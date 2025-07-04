#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for run_iar_2_simulated_data.py.

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
# Note that in single-annotation cases (num_annotators = 1), this hyperparameter is always set to True since otherwise
# IAR 2.0 would not be able to modify the original labels.
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
# Note that in single-annotation cases (num_annotators = 1), the option 'full_agreement' cannot be selected.
validation_loss_criterion = 'full_agreement'

# In single-annotation cases (num_annotators = 1), since we only have one annotator for each frame, if we give
# more weight to the model predictions (i.e. over 0.5), then we are able to make changes to the labels already
# after the first IAR 2.0 iteration. The weight must be in the range [0.0, 1.0]. As an example, a weight of 0.5
# will give equal weight to both the human annotations and the model predictions, a weight of 0.666666 will give
# 2/3 weight to model predictions and 1/3 weight to human annotations, and so on.
# Note that this hyperparameter only applies to single-annotation cases (num_annotators = 1).
model_prediction_weight = 0.666666

# This hyperparameter defines the threshold at which IAR 2.0 is allowed to modify the original soft labels. A value
# of 1.0 means that IAR 2.0 is allowed to modify the labels of all samples in which there was incomplete agreement
# among the annotators, whereas a value of e.g. 0.7 means that IAR 2.0 is only allowed to modify samples in which no
# class had 70% or more of the total probability mass given by the annotators. This hyperparameter can take values
# in the interval [0.0, 1.0], where a lower value means that IAR 2.0 is more conservative in terms of which samples
# it is allowed to modify the labels of. The extreme value of 0.0 means that IAR 2.0 is not allowed to modify any
# labels at all.
# Note that this hyperparameter only affects cases where the number of annotators is more than one (num_annotators > 1).
soft_label_modification_threshold = 1.0

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

# The number of simulated annotators (options: 1, 2, or 3)
num_annotators = 1

# The ratio in which we split our data into training and test sets. For example, a ratio
# of 0.8 will result in 80% of our training data being in the training set and 20% in the test set.
train_test_ratio = 0.8

# The ratio in which we split our training data into training and validation sets. For example, a ratio
# of 0.75 will result in 75% of our training data being in the training set and 25% in the validation set.
train_val_ratio = 0.75

# The number of randomly generated frames of N samples each (N = 120 by default)
num_randomly_generated_frames = 100000

# The number of samples in each simulated signal frame
frame_window_length = 120

# A flag whether we want to add random zero-mean Gaussian noise to the signal
include_noise = True

# The probability of the sinusoid class. The probability of the square wave class will then be
# 1 - sinusoid_class_probability.
sinusoid_class_probability = 0.3

# The noise scale factor for the np.random.normal() function
noise_scale = 0.05

# The probability in which both types of waveforms (sinusoids and square waves) can co-occur within the same frame
class_co_occurrence_probability = 0.9

# The maximum relative amplitude of the co-occurring (distractor) waveform type
co_occurring_class_max_relative_amplitude = 0.99

# The smallest possible amplitude for the generated samples. Note that for sinusoids, this means the smallest
# maximum amplitude.
min_amplitude = 0.2

# The annotation error tendency for annotator 1 (low = 0.1, medium = 0.5, high = 0.8)
annotator_1_error_tendency = 0.5

# The biased class label for annotator 1 (either None, 0 = sinusoid, or 1 = square wave) and the amount of bias towards
# the biased class label (0.0 = no bias, 0.1 = low bias, 0.2 = medium bias, and 0.3 = high bias).
annotator_1_biased_label = 1
annotator_1_bias = 0.2

# We can also change how the annotator errors evolve over time for the simulated annotators, e.g. the annotation error
# might increase gradually if the annotator gets tired. Options: None, 'linear', or 'sawtooth'
annotator_1_error_function = None

# The parameters for our error functions
annotator_1_error_function_params = None

# If we have already created a simulated dataset, select whether we want to load that one (True) or if we want to
# create a new one and overwrite the old one (False)
load_from_file = True

# Select whether we want to shuffle our training data
shuffle_training_data = True

# Select whether we want to shuffle our validation data
shuffle_validation_data = True

# The hyperparameters for our data loaders
params_train_dataset = {'num_annotators': num_annotators,
                        'train_test_ratio': train_test_ratio,
                        'train_val_ratio': train_val_ratio,
                        'num_randomly_generated_frames': num_randomly_generated_frames,
                        'frame_window_length': frame_window_length,
                        'include_noise': include_noise,
                        'sinusoid_class_probability': sinusoid_class_probability,
                        'noise_scale': noise_scale,
                        'class_co_occurrence_probability': class_co_occurrence_probability,
                        'co_occurring_class_max_relative_amplitude': co_occurring_class_max_relative_amplitude,
                        'min_amplitude': min_amplitude,
                        'annotator_1_error_tendency': annotator_1_error_tendency,
                        'annotator_1_biased_label': annotator_1_biased_label,
                        'annotator_1_bias': annotator_1_bias,
                        'annotator_1_error_function': annotator_1_error_function,
                        'annotator_1_error_function_params': annotator_1_error_function_params,
                        'file_save_dir': result_dir,
                        'load_from_file': load_from_file,
                        'file_name': f'simulation_dataset_exp_{experiment_number}.p'}

params_validation_dataset = {'num_annotators': num_annotators,
                             'train_test_ratio': train_test_ratio,
                             'train_val_ratio': train_val_ratio,
                             'num_randomly_generated_frames': num_randomly_generated_frames,
                             'frame_window_length': frame_window_length,
                             'include_noise': include_noise,
                             'sinusoid_class_probability': sinusoid_class_probability,
                             'noise_scale': noise_scale,
                             'class_co_occurrence_probability': class_co_occurrence_probability,
                             'co_occurring_class_max_relative_amplitude': co_occurring_class_max_relative_amplitude,
                             'min_amplitude': min_amplitude,
                             'annotator_1_error_tendency': annotator_1_error_tendency,
                             'annotator_1_biased_label': annotator_1_biased_label,
                             'annotator_1_bias': annotator_1_bias,
                             'annotator_1_error_function': annotator_1_error_function,
                             'annotator_1_error_function_params': annotator_1_error_function_params,
                             'file_save_dir': result_dir,
                             'load_from_file': load_from_file,
                             'file_name': f'simulation_dataset_exp_{experiment_number}.p'}

params_test_dataset = {'num_annotators': num_annotators,
                       'train_test_ratio': train_test_ratio,
                       'train_val_ratio': train_val_ratio,
                       'num_randomly_generated_frames': num_randomly_generated_frames,
                       'frame_window_length': frame_window_length,
                       'include_noise': include_noise,
                       'sinusoid_class_probability': sinusoid_class_probability,
                       'noise_scale': noise_scale,
                       'class_co_occurrence_probability': class_co_occurrence_probability,
                       'co_occurring_class_max_relative_amplitude': co_occurring_class_max_relative_amplitude,
                       'min_amplitude': min_amplitude,
                       'annotator_1_error_tendency': annotator_1_error_tendency,
                       'annotator_1_biased_label': annotator_1_biased_label,
                       'annotator_1_bias': annotator_1_bias,
                       'annotator_1_error_function': annotator_1_error_function,
                       'annotator_1_error_function_params': annotator_1_error_function_params,
                       'file_save_dir': result_dir,
                       'load_from_file': load_from_file,
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
