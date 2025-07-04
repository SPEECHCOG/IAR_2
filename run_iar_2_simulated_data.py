#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for running the IAR 2.0 algorithm for simulated data (binary waveform classification
task with either one, two, or three simulated annotators). After running the IAR 2.0 algorithm,
this code also tests the classification performance of models trained using the original
(discretized) soft labels and the labels produced by the IAR 2.0 algorithm. For single-annotation
cases, the code uses the original labels instead of soft labels (as soft labels are not available).

"""


import numpy as np
import os
import time
import sys
import pickle

from importlib.machinery import SourceFileLoader
from py_conf_file_into_text import convert_py_conf_file_to_text
from sklearn.metrics import f1_score, recall_score, matthews_corrcoef, confusion_matrix
from copy import deepcopy

from torch import cuda, no_grad, save, load, from_numpy
from torch.utils.data import DataLoader
from torch.nn import Softmax


def discretize_categories(Y):
    Y_d = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        vec = Y[i,:] + 0.01*np.random.rand(Y.shape[1])
        ind = np.argmax(vec)
        Y_d[i, ind] = 1.0

    return Y_d


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('\nUsage: \n1) python run_iar_2_simulated_data.py \nOR \n2) python run_iar_2_simulated_data.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try:
        import conf_run_iar_2_simulated_data as conf
        conf_file_name = 'conf_run_iar_2_simulated_data.py'
    except ModuleNotFoundError:
        sys.exit('\nUsage: \n1) python run_iar_2_simulated_data.py \nOR \n2) python run_iar_2_simulated_data.py <configuration_file>\n\n' \
                 'By using the first option, you need to have a configuration file named "conf_run_iar_2_simulated_data.py" in the same ' \
                 'directory as "run_iar_2_simulated_data.py"')

# Import our models
encoder_model = getattr(__import__('iar_2_model', fromlist=[conf.encoder_model]), conf.encoder_model)
timeseries_model = getattr(__import__('iar_2_model', fromlist=[conf.timeseries_model]), conf.timeseries_model)

# Import our dataset for our data loader
dataset = getattr(__import__('iar_2_data_loader', fromlist=[conf.dataset_name]), conf.dataset_name)

# Import our loss function
iar_loss = getattr(__import__('torch.nn', fromlist=[conf.loss_name]), conf.loss_name)

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)

# Import our learning rate scheduler
if conf.use_lr_scheduler:
    scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler]), conf.lr_scheduler)


if __name__ == "__main__":
    
    # We make sure that we are able to write the logging file
    textfile_path, textfile_name = os.path.split(f'{conf.result_dir}/{conf.name_of_log_textfile}')
    if not os.path.exists(textfile_path):
        if textfile_path != '':
            os.makedirs(textfile_path)
    file = open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write(f'Process on {device}\n\n')
    
    # We make modifications to erroneous configuration settings
    if conf.num_annotators == 1:
        if conf.allow_model_to_disagree_with_annotators == False:
            conf.allow_model_to_disagree_with_annotators = True
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('"allow_model_to_disagree_with_annotators" set to True, False is not valid for single-annotation cases!\n')
        if conf.validation_loss_criterion == 'full_agreement':
            conf.validation_loss_criterion = 'original_labels'
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('"validation_loss_criterion" set to "original_labels", "full_agreement" is not valid for single-annotation cases!\n')
    else:
        if conf.soft_label_modification_threshold > 1.0:
            conf.soft_label_modification_threshold = 1.0
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('"soft_label_modification_threshold" set to 1.0, since no class can have a soft label greater than 1.0!\n')
    
    if conf.num_annotators == 1:
        # The weights for the model predictions and human annotations when computing new soft labels
        model_weight = conf.model_prediction_weight
        human_weight = 1 - conf.model_prediction_weight
    
    if conf.num_annotators != 1:
        test_labels_list = ['full agreement labels', 'original labels', 'ground truth labels']
    else:
        test_labels_list = ['original labels', 'ground truth labels']
    
    # We create our dict where the training results are stored
    simulation_results = {}
    for label_index in np.arange(conf.max_iar_iterations + 1):
        if label_index == 0:
            if conf.num_annotators != 1:
                label_name = 'original (discretized) soft labels'
            else:
                label_name = 'original labels'
        else:
            label_name = f'IAR 2.0 labels (iteration {label_index})'
        simulation_results[label_name] = {}
        for test_labels in test_labels_list:
            simulation_results[label_name][test_labels] = []
    
    dataset_file_save_dir = conf.params_train_dataset['file_save_dir']
    dataset_file_name = conf.params_train_dataset['file_name']
    
    for simulation_num in range(1, conf.num_simulation_repeated + 1):
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n')
            f.write(f'Simulation number {simulation_num}\n')
            f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n\n')
    
        # We compute prior probabilities for the given classes for loss weighting
        prior_prob = np.array([conf.sinusoid_class_probability, 1 - conf.sinusoid_class_probability])
        class_weights = from_numpy(1 / prior_prob).to(device).float()
        
        # The soft labels for every IAR 2.0 iteration are stored in a dict
        y_iterations = {}
        iar_iteration = 0
        
        # We create our simulated data (or we load it from a file)
        training_set = dataset(train_val_test='train', **conf.params_train_dataset)
        validation_set = dataset(train_val_test='validation', **conf.params_validation_dataset)
        num_training_samples = len(training_set)
        num_validation_samples = len(validation_set)
        y_iterations[iar_iteration] = np.concatenate((training_set.Y_soft_labels, validation_set.Y_soft_labels), axis=0)
        
        y_discretized = discretize_categories(y_iterations[iar_iteration])
        
        # Save the discretized labels in a separate file
        with open(f'{conf.result_dir}/{conf.label_savename_base}_iteration_{iar_iteration}.p', 'wb') as sv:
            pickle.dump(y_discretized, sv)
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write('################## Running IAR 2.0 ##################\n\n')
        
        iar_iteration += 1
        iar_training_loop = True
        while iar_training_loop:
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'Training model, IAR 2.0 iteration {iar_iteration}\n\n')
            
            # Initialize our models, pass the models to the available device
            Encoder_model = encoder_model(**conf.encoder_model_params).to(device)
            Timeseries_model = timeseries_model(**conf.timeseries_model_params).to(device)
            
            # Give the parameters of our models to an optimizer
            model_parameters = list(Encoder_model.parameters()) + list(Timeseries_model.parameters())
            optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
            
            # Get our learning rate for later use
            learning_rate = optimizer.param_groups[0]['lr']
            
            # Give the optimizer to the learning rate scheduler
            if conf.use_lr_scheduler:
                lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params)
        
            # Initialize our loss function as a class
            if conf.use_class_weights:
                loss_function = iar_loss(weight=class_weights, **conf.loss_params)
            else:
                loss_function = iar_loss(**conf.loss_params)
            
        
            # Variables for early stopping
            highest_validation_accuracy = -1e10
            best_validation_epoch = 0
            patience_counter = 0
            
            # Initialize the best versions of our models
            best_model_encoder = None
            best_model_timeseries = None
            
            if conf.continue_model_training_in_every_iteration and iar_iteration != 1:
                # Load the model weights from the trained model of our previous IAR 2.0 iteration
                try:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('Loading models from file...\n')
                        f.write(f'Loading model {conf.result_dir}/{conf.best_model_encoder_name}...\n')
                        f.write(f'Loading model {conf.result_dir}/{conf.best_model_timeseries_name}...\n')
                    Encoder_model.load_state_dict(load(f'{conf.result_dir}/{conf.best_model_encoder_name}', map_location=device))
                    Timeseries_model.load_state_dict(load(f'{conf.result_dir}/{conf.best_model_timeseries_name}', map_location=device))
                    
                    best_model_encoder = deepcopy(Encoder_model.state_dict())
                    best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('Done!\n\n')
                except FileNotFoundError:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('An error occurred while loading the files! Training with random model weights...\n\n')
            
            elif conf.use_pretrained_model:
                # Load the model weights of a pre-trained model
                try:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('Loading pre-trained models from file...\n')
                        f.write(f'Loading model {conf.pretrained_encoder_name}...\n')
                        f.write(f'Loading model {conf.pretrained_timeseries_name}...\n')
                    Encoder_model.load_state_dict(load(f'{conf.pretrained_encoder_name}', map_location=device))
                    Timeseries_model.load_state_dict(load(f'{conf.pretrained_timeseries_name}', map_location=device))
                    best_model_encoder = deepcopy(Encoder_model.state_dict())
                    best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('Done!\n\n')
                except FileNotFoundError:
                    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                        f.write('An error occurred while loading the files! Training without pretrained model weights...\n\n')
            
            # Initialize our data and labels for the training set data loader
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Initializing training set... ')
            
            # Initialize the data loader for the training set
            training_set = dataset(train_val_test='train', **conf.params_train_dataset)
            y_soft_labels = y_iterations[iar_iteration - 1][:num_training_samples]
            training_set.Y_soft_labels = discretize_categories(y_soft_labels)
            train_data_loader = DataLoader(training_set, **conf.params_train)
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Done!\n')
            
            # Initialize the data loader for the validation set
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Initializing validation set... ')
            
            validation_set = dataset(train_val_test='validation', **conf.params_validation_dataset)
            
            if conf.validation_loss_criterion == 'full_agreement':
                y_soft_labels = y_iterations[iar_iteration - 1][-num_validation_samples:]
            elif conf.validation_loss_criterion == 'original_labels':
                y_soft_labels = y_iterations[0][-num_validation_samples:]
            elif conf.validation_loss_criterion == 'latest_iteration_labels':
                y_soft_labels = y_iterations[iar_iteration - 1][-num_validation_samples:]
            else:
                sys.exit(f'The validation loss criterion {conf.validation_loss_criterion} not implemented!')
            
            validation_set.Y_soft_labels = discretize_categories(y_soft_labels)
            validation_data_loader = DataLoader(validation_set, **conf.params_validation)
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Done!\n\n\n')
            
            # Flag for indicating if the maximum number of training epochs is reached
            max_epochs_reached = 1
            
            # Start training our model
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Starting training...\n')
            
            for epoch in range(1, conf.max_epochs + 1):
                
                start_time = time.time()
        
                # Lists containing the losses of each epoch
                epoch_loss_training = []
                epoch_loss_validation = []
                epoch_true_Y_training = np.array([])
                epoch_pred_Y_training = np.array([])
                epoch_true_Y_validation = np.array([])
                epoch_pred_Y_validation = np.array([])
        
                # Indicate that we are in training mode, so e.g. dropout will function
                Encoder_model.train()
                Timeseries_model.train()
                
                # Loop through every minibatch of our training data
                for train_data in train_data_loader:
                    
                    # Get the minibatches
                    if conf.num_annotators != 1:
                        X, _, _, Y, _ = [element.to(device) for element in train_data]
                    else:
                        X, _, _, Y = [element.to(device) for element in train_data]
                    
                    # Zero the gradient of the optimizer
                    optimizer.zero_grad()
                    
                    # Pass our data through the encoder
                    Embedding = Encoder_model(X.float())
                    
                    # Pass our embeddings to our time-series model
                    Y_pred = Timeseries_model(Embedding)
                    
                    # Compute the loss
                    y_output = Y.max(dim=1)[1]
                    loss = loss_function(input=Y_pred, target=y_output)
                    
                    # Perform the backward pass
                    loss.backward()
                    
                    # Update the weights of our model
                    optimizer.step()
    
                    # Add the loss to the total loss of the batch
                    epoch_loss_training.append(loss.item())
                    
                    # Compute the class predictions of the model
                    smax = Softmax(dim=1)
                    Y_pred_smax_np = smax(Y_pred).detach().cpu().numpy()
                    predictions = np.argmax(Y_pred_smax_np, axis=1)
                    epoch_true_Y_training = np.concatenate((epoch_true_Y_training, y_output.detach().cpu().numpy()))
                    epoch_pred_Y_training = np.concatenate((epoch_pred_Y_training, predictions))
                
                
                # Indicate that we are in evaluation mode, so e.g. dropout will not function
                Encoder_model.eval()
                Timeseries_model.eval()
        
                # Make PyTorch not calculate the gradients, so everything will be much faster.
                with no_grad():
                    
                    # Loop through every minibatch of our validation data and perform a similar process
                    # as for the training data
                    for validation_data in validation_data_loader:
                        if conf.num_annotators != 1:
                            X, _, _, Y, full_agreement_mask = [element.to(device) for element in validation_data]
                        else:
                            X, _, _, Y = [element.to(device) for element in validation_data]
                        Embedding = Encoder_model(X.float())
                        Y_pred = Timeseries_model(Embedding)
                        y_output = Y.max(dim=1)[1]
                        if conf.validation_loss_criterion == 'full_agreement':
                            # 0 = full agreement, 1 = no full agreement so we need to flip the booleans around
                            masked_frames = full_agreement_mask.bool()
                            Y_pred = Y_pred[~masked_frames, :]
                            y_output = y_output[~masked_frames]
                            
                        loss = loss_function(input=Y_pred, target=y_output)
                        epoch_loss_validation.append(loss.item())
                        smax = Softmax(dim=1)
                        Y_pred_smax_np = smax(Y_pred).detach().cpu().numpy()
                        predictions = np.argmax(Y_pred_smax_np, axis=1)
                        epoch_true_Y_validation = np.concatenate((epoch_true_Y_validation, y_output.detach().cpu().numpy()))
                        epoch_pred_Y_validation = np.concatenate((epoch_pred_Y_validation, predictions))
                
                # Calculate mean losses and the prediction accuracy of the model
                epoch_loss_training = np.array(epoch_loss_training).mean()
                epoch_loss_validation = np.array(epoch_loss_validation).mean()
                if conf.train_criterion == 'f1':
                    epoch_accuracy_training = f1_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                    epoch_accuracy_validation = f1_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                elif conf.train_criterion == 'recall':
                    epoch_accuracy_training = recall_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                    epoch_accuracy_validation = recall_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                elif conf.train_criterion == 'mcc':
                    epoch_accuracy_training = matthews_corrcoef(epoch_true_Y_training, epoch_pred_Y_training)
                    epoch_accuracy_validation = matthews_corrcoef(epoch_true_Y_validation, epoch_pred_Y_validation)
                else:
                    sys.exit(f'The training criterion {conf.train_criterion} not implemented!')
                
                # Check early stopping conditions
                if epoch_accuracy_validation > highest_validation_accuracy:
                    highest_validation_accuracy = epoch_accuracy_validation
                    patience_counter = 0
                    best_model_encoder = deepcopy(Encoder_model.state_dict())
                    best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                    best_validation_epoch = epoch
                    if conf.continue_model_training_in_every_iteration:
                        # We first make sure that we are able to write the files
                        save_names = [f'{conf.result_dir}/{conf.best_model_encoder_name}',
                                      f'{conf.result_dir}/{conf.best_model_timeseries_name}']
                        for model_save_name in save_names:
                            model_path, model_filename = os.path.split(model_save_name)
                            if not os.path.exists(model_path):
                                if model_path != '':
                                    os.makedirs(model_path)
                        
                        save(best_model_encoder, f'{conf.result_dir}/{conf.best_model_encoder_name}')
                        save(best_model_timeseries, f'{conf.result_dir}/{conf.best_model_timeseries_name}')
                else:
                    patience_counter += 1
                
                end_time = time.time()
                epoch_time = end_time - start_time
                
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write(f'Epoch: {epoch:04d} | Mean training loss: {epoch_loss_training:6.4f} | '
                      f'Mean validation loss: {epoch_loss_validation:6.4f} | '
                      f'Mean training accuracy: {epoch_accuracy_training:6.4f} | '
                      f'Mean validation accuracy: {epoch_accuracy_validation:6.4f} (highest: {highest_validation_accuracy:6.4f}) | '
                      f'Duration: {epoch_time:4.2f} seconds\n')
                
                # We check that do we need to update the learning rate based on the validation loss
                if conf.use_lr_scheduler:
                    if conf.lr_scheduler == 'ReduceLROnPlateau':
                        lr_scheduler.step(epoch_accuracy_validation)
                    else:
                        lr_scheduler.step()
                    current_learning_rate = optimizer.param_groups[0]['lr']
                    if current_learning_rate != learning_rate:
                        learning_rate = current_learning_rate
                        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                            f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
                
                # If patience counter is fulfilled, we stop the training process
                if patience_counter >= conf.patience:
                    max_epochs_reached = 0
                    break
            
            if max_epochs_reached:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nMax number of epochs reached, stopping training\n\n')
            else:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nExiting due to early stopping\n\n')
            
            if best_model_encoder is None:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nNo best model. The criteria for the lowest acceptable validation accuracy not satisfied!\n\n')
                sys.exit('No best model, exiting...')
            else:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write(f'\nBest epoch {best_validation_epoch} with validation accuracy {highest_validation_accuracy}\n\n')
            
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'\nComputing new soft labels (IAR 2.0 iteration {iar_iteration})... ')
            
            
            # We create a "combined" training and validation set in order to compute new IAR labels
            trainval_set = dataset(train_val_test='train', **conf.params_train_dataset)
            validation_set = dataset(train_val_test='validation', **conf.params_validation_dataset)
            
            if conf.iterate_original_soft_labels:
                # We use the original soft labels for computing the new soft labels
                y_soft_labels = y_iterations[0]
            else:
                # We use the soft labels of the previous iteration for computing the new soft labels
                y_soft_labels = y_iterations[iar_iteration - 1]
            
            trainval_set.X = np.concatenate((trainval_set.X, validation_set.X), axis=0)
            trainval_set.Y_ground_truth = np.concatenate((trainval_set.Y_ground_truth, validation_set.Y_ground_truth), axis=0)
            trainval_set.Y_annotators = np.concatenate((trainval_set.Y_annotators, validation_set.Y_annotators), axis=0)
            trainval_set.Y_soft_labels = y_soft_labels
            if conf.num_annotators != 1:
                trainval_set.full_agreement_mask = np.concatenate((trainval_set.full_agreement_mask, validation_set.full_agreement_mask), axis=0)
            
            trainval_data_loader = DataLoader(trainval_set, **conf.params_trainval)
            
            # We compute new soft labels
            model_output = []
            full_agreement_masks = []
            
            if conf.soft_label_modification_threshold != 1.0:
                annotator_labels = []
            
            # Load the best version of the model
            Encoder_model.load_state_dict(best_model_encoder)
            Timeseries_model.load_state_dict(best_model_timeseries)
            
            Encoder_model.eval()
            Timeseries_model.eval()
    
            with no_grad():
                for trainval_data in trainval_data_loader:
                    if conf.num_annotators != 1:
                        if conf.soft_label_modification_threshold == 1.0:
                            X, _, _, _, full_agreement_mask = [element.to(device) for element in trainval_data]
                        else:
                            X, _, Y_annotators, _, full_agreement_mask = [element.to(device) for element in trainval_data]
                            annotator_labels.append(Y_annotators.detach().cpu().numpy())
                        full_agreement_masks.append(full_agreement_mask.detach().cpu().numpy())
                    else:
                        X, _, _, _ = [element.to(device) for element in trainval_data]
                    Embedding = Encoder_model(X.float())
                    Y_pred = Timeseries_model(Embedding)
                    smax = Softmax(dim=1)
                    Y_pred_smax_np = smax(Y_pred).detach().cpu().numpy()
                    model_output.append(Y_pred_smax_np)
            
            model_output = np.vstack(model_output)
            if conf.num_annotators != 1:
                full_agreement_masks = np.hstack(full_agreement_masks) # 0 = full agreement, 1 = no full agreement
                
                if conf.soft_label_modification_threshold != 1.0:
                    # We don't allow soft label values above a given threshold to be modified during the IAR 2.0
                    # process, so we update full_agreement_masks accordingly.
                    annotator_labels = np.concatenate(annotator_labels, axis=0)
                    class_probabilities = np.stack((1 - annotator_labels.mean(axis=1), annotator_labels.mean(axis=1)), axis=1)
                    class_max_probabilities = class_probabilities.max(axis=1)
                    for i in range(len(class_max_probabilities)):
                        if class_max_probabilities[i] < conf.soft_label_modification_threshold:
                            # We allow for the label to be modified
                            full_agreement_masks[i] = 1.0
                        else:
                            # We are not allowed to modify the given label
                            full_agreement_masks[i] = 0.0
                    
                    
            
            y_new_soft_labels = np.zeros_like(y_soft_labels)
            
            if conf.num_annotators != 1:
                for i in range(len(y_soft_labels)):
                    if full_agreement_masks[i]:
                        # We are allowed to modify the soft label
                        if y_soft_labels[i].sum() == 0:
                            y_new_soft_labels[i] = y_soft_labels[i]
                        else:
                            if conf.allow_model_to_disagree_with_annotators:
                                new_soft_label = (y_soft_labels[i] + model_output[i]) / (y_soft_labels[i] + model_output[i]).sum()
                            else:
                                new_soft_label = (y_soft_labels[i] * model_output[i]) / (y_soft_labels[i] * model_output[i]).sum()
                            y_new_soft_labels[i] = new_soft_label
                    else:
                        # We do not modify the soft label
                        y_new_soft_labels[i] = y_soft_labels[i]
            else:
                # In single-annotation cases, we are allowed to modify the label of all samples
                for i in range(len(y_soft_labels)):
                    if y_soft_labels[i].sum() == 0:
                        y_new_soft_labels[i] = y_soft_labels[i]
                    else:
                        new_soft_label = (human_weight * y_soft_labels[i] + model_weight * model_output[i]) / (human_weight * y_soft_labels[i] + model_weight * model_output[i]).sum()
                        y_new_soft_labels[i] = new_soft_label
            
            y_iterations[iar_iteration] = y_new_soft_labels
            y_discretized = discretize_categories(y_iterations[iar_iteration])
                
            # Save the new discretized labels in a separate file
            with open(f'{conf.result_dir}/{conf.label_savename_base}_iteration_{iar_iteration}.p', 'wb') as sv:
                pickle.dump(y_discretized, sv)
            
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'Done! Labels saved to file {conf.result_dir}/{conf.label_savename_base}_iteration_{iar_iteration}.p...\n\n\n\n')
            
            if iar_iteration > conf.max_iar_iterations - 1:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nMaximum number of IAR 2.0 iterations reached, stopping IAR 2.0 process... ')
                iar_training_loop = False
            iar_iteration += 1
        
        ######################################################################################################
        ######################################################################################################
        ######################################################################################################
        
        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
            f.write('Done!\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            f.write('########################################################################################')
            f.write('########################################################################################\n')
            f.write('########################################################################################')
            f.write('########################################################################################\n')
            f.write('########################################################################################')
            f.write('########################################################################################\n')
            f.write('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
            f.write('Testing model on the test labels...\n\n')
        
        labels_list = []
        
        # Find out all the label files in the given IAR label directory
        try:
            filenames_iar = os.listdir(conf.result_dir)
        except FileNotFoundError:
            sys.exit(f'Given label file directory {conf.result_dir} does not exist!')
        
        # Clean the list if there are other files than Pickle files
        iar_file_names = [filename for filename in filenames_iar if filename.endswith('.p')]
        iar_file_names = sorted(iar_file_names, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        del filenames_iar
        
        # We remove the dataset file name from the list
        try:
            iar_file_names.remove(dataset_file_name)
        except ValueError:
            pass
        
        # We add the original labels and the IAR 2.0 labels of each iteration to a list
        for filename in iar_file_names:
            with open(os.path.join(conf.result_dir, filename), 'rb') as fp:
                iar_labels = pickle.load(fp)
            labels_list.append(iar_labels)
        
        # Start the training process for all of our different label variations
        for label_index in np.arange(len(labels_list)):
            
            if label_index == 0:
                if conf.num_annotators != 1:
                    label_name = 'original (discretized) soft labels'
                else:
                    label_name = 'original labels'
            else:
                label_name = f'IAR 2.0 labels (iteration {label_index})'
            
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write(f'\n\n########## Using {label_name} for training model (label setup {label_index + 1}/{len(labels_list)}) ##########\n\n')
            
            training_set = dataset(train_val_test='train', **conf.params_train_dataset)
            training_set.Y_soft_labels = labels_list[label_index][:num_training_samples]
            train_data_loader = DataLoader(training_set, **conf.params_train)
            
            validation_set = dataset(train_val_test='validation', **conf.params_validation_dataset)
            validation_set.Y_soft_labels = labels_list[label_index][-num_validation_samples:]
            validation_data_loader = DataLoader(validation_set, **conf.params_validation)
            
            # Initialize our models, pass the models to the available device
            Encoder_model = encoder_model(**conf.encoder_model_params).to(device)
            Timeseries_model = timeseries_model(**conf.timeseries_model_params).to(device)
            
            # Give the parameters of our models to an optimizer
            model_parameters = list(Encoder_model.parameters()) + list(Timeseries_model.parameters())
            optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
            
            # Get our learning rate for later use
            learning_rate = optimizer.param_groups[0]['lr']
            
            # Give the optimizer to the learning rate scheduler
            if conf.use_lr_scheduler:
                lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params)
        
            # Initialize our loss function as a class
            if conf.use_class_weights:
                loss_function = iar_loss(weight=class_weights, **conf.loss_params)
            else:
                loss_function = iar_loss(**conf.loss_params)
            
            # Flag for indicating if the maximum number of training epochs is reached
            max_epochs_reached = 1
            
            # Variables for early stopping
            highest_validation_accuracy = -1e10
            best_validation_epoch = 0
            patience_counter = 0
            
            # Initialize the best versions of our models
            best_model_encoder = None
            best_model_timeseries = None
            
            # Start training our model in a similar manner as in the IAR 2.0 process
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('Starting training...\n')
            
            for epoch in range(1, conf.max_epochs + 1):
                
                start_time = time.time()
        
                epoch_loss_training = []
                epoch_loss_validation = []
                epoch_true_Y_training = np.array([])
                epoch_pred_Y_training = np.array([])
                epoch_true_Y_validation = np.array([])
                epoch_pred_Y_validation = np.array([])
        
                Encoder_model.train()
                Timeseries_model.train()
                
                for train_data in train_data_loader:
                    if conf.num_annotators != 1:
                        X, _, _, Y, _ = [element.to(device) for element in train_data]
                    else:
                        X, _, _, Y = [element.to(device) for element in train_data]
                    optimizer.zero_grad()
                    Embedding = Encoder_model(X.float())
                    Y_pred = Timeseries_model(Embedding)
                    y_output = Y.max(dim=1)[1]
                    loss = loss_function(input=Y_pred, target=y_output)
                    loss.backward()
                    optimizer.step()
                    epoch_loss_training.append(loss.item())
                    smax = Softmax(dim=1)
                    Y_pred_smax_np = smax(Y_pred).detach().cpu().numpy()
                    predictions = np.argmax(Y_pred_smax_np, axis=1)
                    epoch_true_Y_training = np.concatenate((epoch_true_Y_training, y_output.detach().cpu().numpy()))
                    epoch_pred_Y_training = np.concatenate((epoch_pred_Y_training, predictions))
                
                Encoder_model.eval()
                Timeseries_model.eval()
        
                with no_grad():
                    for validation_data in validation_data_loader:
                        if conf.num_annotators != 1:
                            X, _, _, Y, _ = [element.to(device) for element in validation_data]
                        else:
                            X, _, _, Y = [element.to(device) for element in validation_data]
                        Embedding = Encoder_model(X.float())
                        Y_pred = Timeseries_model(Embedding)
                        y_output = Y.max(dim=1)[1]
                        loss = loss_function(input=Y_pred, target=y_output)
                        epoch_loss_validation.append(loss.item())
                        smax = Softmax(dim=1)
                        Y_pred_smax_np = smax(Y_pred).detach().cpu().numpy()
                        predictions = np.argmax(Y_pred_smax_np, axis=1)
                        epoch_true_Y_validation = np.concatenate((epoch_true_Y_validation, y_output.detach().cpu().numpy()))
                        epoch_pred_Y_validation = np.concatenate((epoch_pred_Y_validation, predictions))
                
                epoch_loss_training = np.array(epoch_loss_training).mean()
                epoch_loss_validation = np.array(epoch_loss_validation).mean()
                if conf.train_criterion == 'f1':
                    epoch_accuracy_training = f1_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                    epoch_accuracy_validation = f1_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                elif conf.train_criterion == 'recall':
                    epoch_accuracy_training = recall_score(epoch_true_Y_training, epoch_pred_Y_training, average='macro')
                    epoch_accuracy_validation = recall_score(epoch_true_Y_validation, epoch_pred_Y_validation, average='macro')
                elif conf.train_criterion == 'mcc':
                    epoch_accuracy_training = matthews_corrcoef(epoch_true_Y_training, epoch_pred_Y_training)
                    epoch_accuracy_validation = matthews_corrcoef(epoch_true_Y_validation, epoch_pred_Y_validation)
                else:
                    sys.exit(f'The training criterion {conf.train_criterion} not implemented!')
                
                if epoch_accuracy_validation > highest_validation_accuracy:
                    highest_validation_accuracy = epoch_accuracy_validation
                    patience_counter = 0
                    best_model_encoder = deepcopy(Encoder_model.state_dict())
                    best_model_timeseries = deepcopy(Timeseries_model.state_dict())
                    best_validation_epoch = epoch
                else:
                    patience_counter += 1
                
                end_time = time.time()
                epoch_time = end_time - start_time
                
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write(f'Epoch: {epoch:04d} | Mean training loss: {epoch_loss_training:6.4f} | '
                      f'Mean validation loss: {epoch_loss_validation:6.4f} | '
                      f'Mean training accuracy: {epoch_accuracy_training:6.4f} | '
                      f'Mean validation accuracy: {epoch_accuracy_validation:6.4f} (highest: {highest_validation_accuracy:6.4f}) | '
                      f'Duration: {epoch_time:4.2f} seconds\n')
                
                if conf.use_lr_scheduler:
                    if conf.lr_scheduler == 'ReduceLROnPlateau':
                        lr_scheduler.step(epoch_accuracy_validation)
                    else:
                        lr_scheduler.step()
                    current_learning_rate = optimizer.param_groups[0]['lr']
                    if current_learning_rate != learning_rate:
                        learning_rate = current_learning_rate
                        with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                            f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
                
                if patience_counter >= conf.patience:
                    max_epochs_reached = 0
                    break
            
            if max_epochs_reached:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nMax number of epochs reached, stopping training\n\n')
            else:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nExiting due to early stopping\n\n')
            
            if best_model_encoder is None:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write('\nNo best model. The criteria for the lowest acceptable validation accuracy not satisfied!\n\n')
                sys.exit('No best model, exiting...')
            else:
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write(f'\nBest epoch {best_validation_epoch} with validation accuracy {highest_validation_accuracy}\n\n')
            
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('\nTesting using different labels...\n')
            
            output_categories = 2
            
            # Test our model using different test labels
            for test_labels in test_labels_list:
                test_set = dataset(train_val_test='test', **conf.params_test_dataset)
                if test_labels != 'ground truth labels':
                    test_set.Y_soft_labels = discretize_categories(test_set.Y_soft_labels)
                test_data_loader = DataLoader(test_set, **conf.params_test)
                
                with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                    f.write(f'Starting testing ({test_labels}), saving results into dict...\n')
                
                Encoder_model.load_state_dict(best_model_encoder)
                Timeseries_model.load_state_dict(best_model_timeseries)
                Encoder_model.eval()
                Timeseries_model.eval()
                
                epoch_true_Y_testing = np.array([])
                epoch_pred_Y_testing = np.array([])
        
                with no_grad():
                    for test_data in test_data_loader:
                        if test_labels == 'ground truth labels':
                            if conf.num_annotators != 1:
                                X, y_output, _, _, _ = [element.to(device) for element in validation_data]
                            else:
                                X, y_output, _, _ = [element.to(device) for element in validation_data]
                        else:
                            if conf.num_annotators != 1:
                                X, _, _, Y, full_agreement_mask = [element.to(device) for element in validation_data]
                            else:
                                X, _, _, Y = [element.to(device) for element in validation_data]
                            y_output = Y.max(dim=1)[1]
                        Embedding = Encoder_model(X.float())
                        Y_pred = Timeseries_model(Embedding)
                        if test_labels == 'full agreement labels':
                            # 0 = full agreement, 1 = no full agreement so we need to flip the booleans around
                            masked_frames = full_agreement_mask.bool()
                            Y_pred = Y_pred[~masked_frames, :]
                            y_output = y_output[~masked_frames]
                        smax = Softmax(dim=1)
                        Y_pred_smax_np = smax(Y_pred).detach().cpu().numpy()
                        predictions = np.argmax(Y_pred_smax_np, axis=1)
                        epoch_true_Y_testing = np.concatenate((epoch_true_Y_testing, y_output.detach().cpu().numpy()))
                        epoch_pred_Y_testing = np.concatenate((epoch_pred_Y_testing, predictions))
                    conf_mat = confusion_matrix(epoch_true_Y_testing, epoch_pred_Y_testing, labels=np.arange(output_categories))
                
                prec = np.zeros(output_categories)
                rec = np.zeros(output_categories)
                f1 = np.zeros(output_categories)
                for i in range(output_categories):
                    # Check if target contains current category
                    containsCat = np.sum(conf_mat[:,i]) > 0
                    if containsCat:
                        prec[i] = np.float32(conf_mat[i,i])/np.float32(np.sum(conf_mat[i,:]))
                        rec[i] = np.float32(conf_mat[i,i])/np.float32(np.sum(conf_mat[:,i]))
                        if np.isnan(prec[i]):
                            prec[i] = 0.0
                        if np.isnan(rec[i]):
                            rec[i] = 0.0    
                        f1[i] = 2.0*prec[i]*rec[i]/(prec[i] + rec[i])
               
                        if np.isnan(f1[i]):
                            f1[i] = 0.0        
                    else:
                        prec[i] = np.nan; rec[i] = np.nan; f1[i] = np.nan
        
                f1_mean = np.nanmean(f1)
                simulation_results[label_name][test_labels].append(f1_mean)
        
        # We remove the old dataset file in order to give way to a new one
        if os.path.exists(os.path.join(dataset_file_save_dir, dataset_file_name)):
            os.remove(os.path.join(dataset_file_save_dir, dataset_file_name))
        else:
            sys.exit('Something is wrong with deleting the old dataset file!')
        
        # We also remove the old label files
        for filename in iar_file_names:
            if os.path.exists(os.path.join(conf.result_dir, filename)):
                os.remove(os.path.join(conf.result_dir, filename))
            else:
                sys.exit('Something is wrong with deleting the old label files!')
        
    
    # We print the results
    for label_index in np.arange(conf.max_iar_iterations + 1):
        if label_index == 0:
            if conf.num_annotators != 1:
                label_name = 'original (discretized) soft labels'
            else:
                label_name = 'original labels'
        else:
            label_name = f'IAR 2.0 labels (iteration {label_index})'
        for test_labels in test_labels_list:
            result = np.array(simulation_results[label_name][test_labels]).mean()
    
            with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
                f.write('\n########################################################################################\n')
                f.write(f'Mean F1 score of {conf.num_simulation_repeated} simulations, with {label_name} as ')
                f.write(f'training labels and {test_labels} as test labels: {result}')
                f.write('\n########################################################################################\n')
        
    with open(f'{conf.result_dir}/{conf.name_of_log_textfile}', 'a') as f:
        f.write('\n\nAll experiments completed!\n')
        
    
    
        
        
        
        

    
