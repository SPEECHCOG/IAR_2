#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for train_iar_2_model_simulated_data.py.

"""

experiment_number = 1

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True

# The directory where the experimental results are saved (the models and the logging output)
result_dir = f'simulation_results_{experiment_number}'

# The base name of the file of the IAR 2.0 labels for each IAR 2.0 iteration. Please note that this file
# (and its potential directory) will be saved under the directory result_dir. After the file name,
# the number of the IAR 2.0 iteration and ".p" (Pickle file) will be added automatically.
label_savename_base = f'simulation_experiment_{experiment_number}_iar_2_labels'

# The name of the text file into which we log the output of the training process. Please note that this
# file (and its potential directory) will be saved under the directory result_dir.
name_of_log_textfile = f'iar_2_simulation_trainlog_{experiment_number}.txt'

# The names of the model weight files of the best models (according to validation accuracy) for
# loading/saving model weights. Please note that these files (and directories) will be saved
# under the directory result_dir under training fold-specific directories.
best_model_encoder_name = 'saved_models/iar_2_best_encoder_model.pt'
best_model_timeseries_name = 'saved_models/iar_2_best_timeseries_model.pt'

# The names of the model weight files of pre-trained models. This setting will only apply if "use_pretrained_model"
# is set to True.
pretrained_encoder_name = 'pretrained_models/pretrained_encoder_model.pt'
pretrained_timeseries_name = 'pretrained_models/pretrained_timeseries_model.pt'

# The number of times the simulation experiment is repeated
num_simulation_repeated = 3


"""The hyperparameters for our training process"""

# The maximum number of training epochs
max_epochs = 800

# The maximum number of IAR 2.0 iterations
max_iar_iterations = 3

# Flag for determining whether we want to use the original soft labels for computing new soft labels in each
# IAR 2.0 iteration, or if we want to use the soft labels of the previous iteration for computing  new soft labels.
#     True: new_soft_label = original_soft_label * model_prediction (normalized to sum up to 1)
#     False: new_soft_label = previous_iar_iteration_soft_label * model prediction (normalized to sum up to 1)
iterate_original_soft_labels = False

# Flag for determining whether we allow the model to suggest labels that have not gotten any soft label probability
# by the annotators. Setting to True turns the multiplication into a sum when computing new soft labels, i.e.
#     new_label = previous_label * model_prediction  -->  new_label = previous_label + model_prediction
allow_model_to_disagree_with_annotators = True

# Flag for determining whether we want to have a randomly initialized model at the start of every IAR 2.0 iteration (False),
# or whether we want to use the weights of the previous IAR iteration's trained model as our starting point for the
# next IAR 2.0 iteration (True)
continue_model_training_in_every_iteration = False

# Flag for determining whether we want to use pre-trained models in our IAR 2.0 process (models are initialized with
# pre-training weights at the start of every IAR 2.0 iteration). # Please note that if
# "continue_model_training_in_every_iteration" is set to True and "use_pretrained_model" is also set to True,
# pre-trained models are only used in the first IAR 2.0 iteration.
use_pretrained_model = False

# The criterion to select which labels to use for computing the validation loss. Options:
#     'full_agreement': use the labels in which all annotators agreed on the label
#     'original_labels': use the original (discretized) labels
#     'latest_iteration_labels': use the IAR labels (discretized) of the previous iteration
validation_loss_criterion = 'full_agreement'

# The learning rate of our model training
learning_rate = 1e-4

# The number of input sequences that we feed into our model before computing the mean loss (and performing
# backpropagation during training).
batch_size = 128

# The patience counter for early stopping
patience = 100

# Dropout rate of the encoder model
dropout_encoder_model = 0.1

# Dropout rate of the timeseries model
dropout_timeseries_model = 0.1

# Select the training criterion
train_criterion = 'f1' # Options: 'f1' / 'recall' / 'mcc'

# Define our loss function that we want to use from torch.nn
loss_name = 'CrossEntropyLoss'

# The hyperparameters for the loss function
loss_params = {}

# Defines whether we want to use class weighting in our loss based on probability priors
use_class_weights = False

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'Adam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = True

# Define which learning rate scheduler we want to use from torch.optim.lr_scheduler
lr_scheduler = 'ReduceLROnPlateau'

# The hyperparameters for the learning rate scheduler
lr_scheduler_params = {'mode': 'max',
                       'factor': 0.5,
                       'patience': 30}


"""The hyperparameters for our dataset and data loaders"""

# Define our dataset for our data loader that we want to use from the file iar_2_data_loader.py
dataset_name = 'simulated_1d_dataset_with_annotators'

# The ratio in which we split our data into training and test sets. For example, a ratio
# of 0.8 will result in 80% of our training data being in the training set and 20% in the test set.
train_test_ratio = 0.8

# The ratio in which we split our training data into training and validation sets. For example, a ratio
# of 0.75 will result in 75% of our training data being in the training set and 25% in the validation set.
train_val_ratio = 0.75

# The number of randomly generated frames of N samples each (N = 120 by default)
num_randomly_generated_frames = 100000

# The probability of the sinusoid class. The probability of the square wave class will then be
# 1 - sinusoid_class_probability.
sinusoid_class_probability = 0.1

# The noise scale for the np.random.normal() function
noise_scale = 0.05

# The probability in which both types of waveforms (sinusoids and square waves) can co-occur within the same frame
class_co_occurrence_probability = 0.9

# The smallest possible amplitude for the generated samples. Note that for sinusoids, this means the smallest
# maximum amplitude.
min_amplitude = 0.2

# The annotation error tendency for annotator 1 (low = 0.1, medium = 0.5, high = 0.8)
annotator_1_error_tendency = 0.1

# The annotation error tendency for annotator 2 (low = 0.1, medium = 0.5, high = 0.8)
annotator_2_error_tendency = 0.8

# The annotation error tendency for annotator 3 (low = 0.1, medium = 0.5, high = 0.8)
annotator_3_error_tendency = 0.8

# The biased class label for annotator 1 (either None, 0 = sinusoid, or 1 = square wave) and the amount of bias towards
# the biased class label (0.0 = no bias, 0.1 = low bias, 0.2 = medium bias, and 0.3 = high bias).
annotator_1_biased_label = None
annotator_1_bias = 0.0

# The biased class label for annotator 2 (either None, 0 = sinusoid, or 1 = square wave) and the amount of bias towards
# the biased class label (0.0 = no bias, 0.1 = low bias, 0.2 = medium bias, and 0.3 = high bias).
annotator_2_biased_label = 1
annotator_2_bias = 0.1

# The biased class label for annotator 3 (either None, 0 = sinusoid, or 1 = square wave) and the amount of bias towards
# the biased class label (0.0 = no bias, 0.1 = low bias, 0.2 = medium bias, and 0.3 = high bias).
annotator_3_biased_label = None
annotator_3_bias = 0.0

# Select whether we want to shuffle our training data
shuffle_training_data = True

# Select whether we want to shuffle our validation data
shuffle_validation_data = True

# The hyperparameters for our data loaders
params_train_dataset = {'train_test_ratio': train_test_ratio,
                        'train_val_ratio': train_val_ratio,
                        'num_randomly_generated_frames': num_randomly_generated_frames,
                        'sinusoid_class_probability': sinusoid_class_probability,
                        'noise_scale': noise_scale,
                        'class_co_occurrence_probability': class_co_occurrence_probability,
                        'min_amplitude': min_amplitude,
                        'annotator_1_error_tendency': annotator_1_error_tendency,
                        'annotator_2_error_tendency': annotator_2_error_tendency,
                        'annotator_3_error_tendency': annotator_3_error_tendency,
                        'annotator_1_biased_label': annotator_1_biased_label,
                        'annotator_1_bias': annotator_1_bias,
                        'annotator_2_biased_label': annotator_2_biased_label,
                        'annotator_2_bias': annotator_2_bias,
                        'annotator_3_biased_label': annotator_3_biased_label,
                        'annotator_3_bias': annotator_3_bias,
                        'file_save_dir': result_dir,
                        'file_name': f'simulation_dataset_exp_{experiment_number}.p'}

params_validation_dataset = {'train_test_ratio': train_test_ratio,
                             'train_val_ratio': train_val_ratio,
                             'num_randomly_generated_frames': num_randomly_generated_frames,
                             'sinusoid_class_probability': sinusoid_class_probability,
                             'noise_scale': noise_scale,
                             'class_co_occurrence_probability': class_co_occurrence_probability,
                             'min_amplitude': min_amplitude,
                             'annotator_1_error_tendency': annotator_1_error_tendency,
                             'annotator_2_error_tendency': annotator_2_error_tendency,
                             'annotator_3_error_tendency': annotator_3_error_tendency,
                             'annotator_1_biased_label': annotator_1_biased_label,
                             'annotator_1_bias': annotator_1_bias,
                             'annotator_2_biased_label': annotator_2_biased_label,
                             'annotator_2_bias': annotator_2_bias,
                             'annotator_3_biased_label': annotator_3_biased_label,
                             'annotator_3_bias': annotator_3_bias,
                             'file_save_dir': result_dir,
                             'file_name': f'simulation_dataset_exp_{experiment_number}.p'}

params_test_dataset = {'train_test_ratio': train_test_ratio,
                       'train_val_ratio': train_val_ratio,
                       'num_randomly_generated_frames': num_randomly_generated_frames,
                       'sinusoid_class_probability': sinusoid_class_probability,
                       'noise_scale': noise_scale,
                       'class_co_occurrence_probability': class_co_occurrence_probability,
                       'min_amplitude': min_amplitude,
                       'annotator_1_error_tendency': annotator_1_error_tendency,
                       'annotator_2_error_tendency': annotator_2_error_tendency,
                       'annotator_3_error_tendency': annotator_3_error_tendency,
                       'annotator_1_biased_label': annotator_1_biased_label,
                       'annotator_1_bias': annotator_1_bias,
                       'annotator_2_biased_label': annotator_2_biased_label,
                       'annotator_2_bias': annotator_2_bias,
                       'annotator_3_biased_label': annotator_3_biased_label,
                       'annotator_3_bias': annotator_3_bias,
                       'file_save_dir': result_dir,
                       'file_name': f'simulation_dataset_exp_{experiment_number}.p'}

# The hyperparameters for training, validation, and testing (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': shuffle_training_data,
                'drop_last': False}

params_validation = {'batch_size': batch_size,
                     'shuffle': shuffle_validation_data,
                     'drop_last': False}

params_trainval = {'batch_size': batch_size,
                   'shuffle': False,
                   'drop_last': False}

params_test = {'batch_size': batch_size,
               'shuffle': False,
               'drop_last': False}

"""The neural network hyperparameters"""

# The names of our encoder and time-series models from the file iar_2_model.py
encoder_model = 'framed_signal_encoder_CNN'
timeseries_model = 'MLP_classifier'

# The hyperparameters for constructing the encoder model. An empty dictionary will make the model to use
# only default hyperparameters

# The number of encoder input channels
encoder_num_input_channels = 1

# The number of encoder output channels
encoder_num_output_channels = 128

# Hyperparameters for our encoder model
encoder_model_params = {'conv_1_in_dim': encoder_num_input_channels,
                        'conv_1_out_dim': 128,
                        'num_norm_features_1': 128,
                        'conv_2_in_dim': 128,
                        'conv_2_out_dim': 128,
                        'num_norm_features_2': 128,
                        'conv_3_in_dim': 128,
                        'conv_3_out_dim': 128,
                        'num_norm_features_3': 128,
                        'conv_4_in_dim': 128,
                        'conv_4_out_dim': encoder_num_output_channels,
                        'num_norm_features_4': encoder_num_output_channels,
                        'dropout': dropout_encoder_model}

timeseries_model_params = {}