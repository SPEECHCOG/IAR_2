# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The data loader for the simulated IAR 2.0 experiments.

"""

import numpy as np
from torch.utils.data import Dataset
import os
import sys
import pickle
import scipy




class simulated_1d_dataset_with_annotators(Dataset):
    """
    A simulated dataset of 1D time-domain data, containing two classes: sinusoids and square waves. For
    each simulated sample, there are three simulated annotators, each with its unique annotation error
    tendency and annotation bias.
    
    Annotator error tendency should be in the range [0,1].
      Annotator_error_tendency = 0.0 --> very low chance of making errors
      Annotator_error_tendency = 0.5 --> normal chance of making errors
      Annotator_error_tendency = 1.0 --> very high chance of making errors
    
    Also, annotator bias should be in the range [0,1].
      Annotator_bias = 0.0 --> there is no bias towards either of the class labels
      Annotator_bias = 1.0 --> the annotator is fully biased towards the other class label, and
                               never gives the other label
    
    """

    def __init__(self, train_val_test = 'train', train_test_ratio = 0.8, train_val_ratio = 0.75, random_seed = 42,
                 file_save_dir = './simulation_datasets', load_from_file = True, file_name = None,
                 num_randomly_generated_frames = 100000, frame_window_length = 120, sinusoid_class_probability = 0.1,
                 include_noise = True, noise_scale = 0.05, class_co_occurrence_probability = 0.9,
                 co_occurring_class_max_relative_amplitude = 0.99, min_amplitude = 0.2,
                 annotator_1_error_tendency = 0.1, annotator_2_error_tendency = 0.2, annotator_3_error_tendency = 0.0,
                 annotator_1_biased_label = None, annotator_1_bias = 0.0, annotator_2_biased_label = None,
                 annotator_2_bias = 0.0, annotator_3_biased_label = None, annotator_3_bias = 0.0,
                 annotator_1_error_function = None, annotator_2_error_function = None,
                 annotator_3_error_function = None, annotator_1_error_function_params = None,
                 annotator_2_error_function_params = None, annotator_3_error_function_params = None):
        super().__init__()
        
        if file_name is None:
            filename = (f'simulated_1d_dataset_{num_randomly_generated_frames}_{frame_window_length}_'
                        f'{sinusoid_class_probability}_{include_noise}_{noise_scale}_{class_co_occurrence_probability}_'
                        f'{co_occurring_class_max_relative_amplitude}_{min_amplitude}_{annotator_1_error_tendency}_'
                        f'{annotator_2_error_tendency}_{annotator_3_error_tendency}_{annotator_1_biased_label}_'
                        f'{annotator_1_bias}_{annotator_2_biased_label}_{annotator_2_bias}_'
                        f'{annotator_3_biased_label}_{annotator_3_bias}_{annotator_1_error_function}_'
                        f'{annotator_2_error_function}_{annotator_3_error_function}').replace('.', '') + '.p'
        else:
            filename = file_name
        
        if not os.path.exists(file_save_dir):
            os.makedirs(file_save_dir)
            load_from_file = False
        else:
            if load_from_file:
                try:
                    with open(os.path.join(file_save_dir, filename), 'rb') as fp:
                        data_dict = pickle.load(fp)
                except FileNotFoundError:
                    load_from_file = False
        
        if not load_from_file:
            
            data_dict = {}
            
            annotator_error_probs = []
            annotator_error_values = [annotator_1_error_tendency, annotator_2_error_tendency, annotator_3_error_tendency]
            annotator_biased_labels = [annotator_1_biased_label, annotator_2_biased_label, annotator_3_biased_label]
            annotator_biases = [annotator_1_bias, annotator_2_bias, annotator_3_bias]
            annotator_error_functions = [annotator_1_error_function, annotator_2_error_function,
                                         annotator_3_error_function]
            annotator_error_function_params = [annotator_1_error_function_params, annotator_2_error_function_params,
                                               annotator_3_error_function_params]
            for i in range(len(annotator_error_functions)):
                if annotator_error_functions[i] is not None:
                    annotator_error_probs.append(self.compute_annotator_error_probs(annotator_error_functions[i],
                                                                                    annotator_error_function_params[i],
                                                                                    num_randomly_generated_frames))
                else:
                    annotator_error_probs.append(annotator_error_values[i])
            
            Data = []
            Y_annotators = []
            Y_ground_truth = []
            for frame_index in range(num_randomly_generated_frames):
                
                Y_sample = []
                
                signal_phase = np.random.uniform(low=-5.0, high=5.0)
                signal_frequency = np.random.uniform(low=0.05, high=0.5)
                signal_amplitude = np.random.rand()
                while signal_amplitude < min_amplitude:
                    signal_amplitude = np.random.rand()
                overlap = False
                amplitude_ratio = 0.0
                
                if np.random.rand() < class_co_occurrence_probability:
                    overlap = True
                    overlap_phase = np.random.uniform(low=-5.0, high=5.0)
                    overlap_frequency = np.random.uniform(low=0.05, high=0.5)
                    overlap_amplitude = co_occurring_class_max_relative_amplitude * np.random.rand() * signal_amplitude
                    amplitude_ratio = overlap_amplitude / signal_amplitude
                    
                    # We want the overlapped signals to be within the range [0,1]
                    overlap_scaler = co_occurring_class_max_relative_amplitude + 1
                
                x = np.linspace(0, frame_window_length, frame_window_length)
                
                if np.random.rand() < sinusoid_class_probability:
                    signal = signal_amplitude * np.sin(signal_frequency * x + signal_phase)
                    ground_truth_label = 0
                    Y_ground_truth.append(ground_truth_label)
                    if overlap:
                        signal = (signal + overlap_amplitude * scipy.signal.square(overlap_frequency * x + overlap_phase)) / overlap_scaler
                else:
                    signal = signal_amplitude * scipy.signal.square(signal_frequency * x + signal_phase)
                    ground_truth_label = 1
                    Y_ground_truth.append(ground_truth_label)
                    if overlap:
                        signal = (signal + overlap_amplitude * np.sin(overlap_frequency * x + overlap_phase)) / overlap_scaler
                
                for i in range(len(annotator_error_probs)):
                    if type(annotator_error_probs[i]) == float:
                        annotator_error_prob = annotator_error_probs[i]
                    else:
                        annotator_error_prob = annotator_error_probs[i][frame_index]
                    
                    annotator_label = self.compute_annotator_label(annotator_error_prob, amplitude_ratio, ground_truth_label,
                                                                   annotator_biased_labels[i], annotator_biases[i])
                    Y_sample.append(annotator_label)
                
                if include_noise:
                    signal = signal + np.random.normal(scale=noise_scale, size=len(x))
            
                Data.append(signal)
                Y_annotators.append(Y_sample)
            
            # We compute the full agreement mask for validation loss computation (0 = full agreement, 1 = no full agreement)
            Y_annotators = np.array(Y_annotators)
            full_agreement_mask = np.ones(len(Y_annotators))
            for i in range(len(Y_annotators)):
                if np.all(Y_annotators[i, :] == Y_annotators[i, 0]):
                    full_agreement_mask[i] = 0
            
            # We compute the one-hot labels and soft labels
            Y_soft_labels = []
            for i in range(len(Y_annotators)):
                class_1_prob = Y_annotators[i, :].mean()
                Y_soft_labels.append([1 - class_1_prob, class_1_prob])
            
            data_dict['X'] = np.array(Data)
            data_dict['Y_ground_truth'] = np.array(Y_ground_truth)
            data_dict['Y_annotators'] = Y_annotators
            data_dict['Y_soft_labels'] = np.array(Y_soft_labels)
            data_dict['full_agreement_mask'] = full_agreement_mask
            
            with open(os.path.join(file_save_dir, filename), 'wb') as sv:
                pickle.dump(data_dict, sv)
        
        # Split our data into a train, validation, and test set
        trainval_data = {}
        train_data = {}
        val_data = {}
        test_data = {}
        np.random.seed(random_seed)
        mask_traintest_split = np.random.rand(len(data_dict['X'])) <= train_test_ratio
        for name in ['X', 'Y_ground_truth', 'Y_annotators', 'Y_soft_labels', 'full_agreement_mask']:
            trainval_data[name] = data_dict[name][mask_traintest_split]
            test_data[name] = data_dict[name][~mask_traintest_split]
        np.random.seed(2*random_seed)    # We use a different random seed for splitting trainval_files
        mask_trainval_split = np.random.rand(len(trainval_data['X'])) <= train_val_ratio
        for name in ['X', 'Y_ground_truth', 'Y_annotators', 'Y_soft_labels', 'full_agreement_mask']:
            train_data[name] = trainval_data[name][mask_trainval_split]
            val_data[name] = trainval_data[name][~mask_trainval_split]
        
        del data_dict
        del trainval_data
        
        if train_val_test == 'train':
            self.X = train_data['X']
            self.Y_ground_truth = train_data['Y_ground_truth']
            self.Y_annotators = train_data['Y_annotators']
            self.Y_soft_labels = train_data['Y_soft_labels']
            self.full_agreement_mask = train_data['full_agreement_mask']
        elif train_val_test == 'validation':
            self.X = val_data['X']
            self.Y_ground_truth = val_data['Y_ground_truth']
            self.Y_annotators = val_data['Y_annotators']
            self.Y_soft_labels = val_data['Y_soft_labels']
            self.full_agreement_mask = val_data['full_agreement_mask']
        else:
            self.X = test_data['X']
            self.Y_ground_truth = test_data['Y_ground_truth']
            self.Y_annotators = test_data['Y_annotators']
            self.Y_soft_labels = test_data['Y_soft_labels']
            self.full_agreement_mask = test_data['full_agreement_mask']
    
    
    
    def compute_annotator_label(self, annotator_error_prob, amplitude_ratio, ground_truth_label,
                                annotator_biased_label, annotator_bias):
        
        """
        Here we calculate an error probability based on the sigmoid function. On top of that, we
        also add a bias term b to the computation in the following way:
            If the bias b is nonzero and there is bias towards one label over the other:
              If the ground truth label is the label with the bias:
                We reduce b from error_prob --> smaller chance for annotation error (annotator favours biased label)
              If the ground truth label is not the label with the bias:
                We add b to error_prob --> higher chance for annotation error (annotator favours the other label)
        """
        
        error_multiplier = 10 * (1 / 10)**annotator_error_prob
        error_prob = 1 / (1 + np.exp(-error_multiplier*(amplitude_ratio-1.0))) # sigmoid function whose center is at the value x = 1.0
        
        if annotator_biased_label is not None:
            if annotator_biased_label not in [0, 1]:
                sys.exit(f'The biased label should be either 0 or 1, and not {annotator_biased_label}!')
            if annotator_biased_label == ground_truth_label:
                error_prob = error_prob - annotator_bias
            else:
                error_prob = error_prob + annotator_bias
        
        if np.random.rand() < error_prob:
            # We add a wrong label
            if ground_truth_label == 0:
                annotator_label = 1
            else:
                annotator_label = 0
        else:
            # we add the correct label
            annotator_label = ground_truth_label
        
        return annotator_label
    
    
    def compute_annotator_error_probs(self, error_function, error_function_params, num_frames):
        
        
        if error_function == 'linear':
            annotator_error_probs = np.linspace(error_function_params['start_value'],
                                                error_function_params['stop_value'], num_frames)
        elif error_function == 'sawtooth':
            x = np.linspace(0, 1, num_frames)
            num_sawtooths = error_function_params['num_sawtooths']
            sawtooth_max_value = error_function_params['sawtooth_max_value']
            annotator_error_probs = sawtooth_max_value * 0.5 * (scipy.signal.sawtooth(2 * np.pi * num_sawtooths * x) + 1)
        else:
            sys.exit(f'{error_function} is a wrong value for the error function name!')
        
        return annotator_error_probs
        
    
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        
        return self.X[index], self.Y_ground_truth[index], self.Y_annotators[index], self.Y_soft_labels[index], self.full_agreement_mask[index]