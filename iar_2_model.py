#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The different models used in the IAR 2.0 simulation experiments.

"""


import sys
from torch.nn import Module, Conv1d, Dropout, AvgPool1d
from torch.nn import GELU, ReLU, LayerNorm, Linear, BatchNorm1d, Identity, ELU




class framed_signal_encoder_CNN(Module):
    """
    A four-layer CNN (containing strided convolutions) for framed raw signal data.
    
    """
    
    def __init__(self,
                 conv_1_in_dim = 1,
                 conv_1_out_dim = 128,
                 conv_1_kernel_size = 10,
                 conv_1_stride = 5,
                 conv_1_padding = 3,
                 num_norm_features_1 = 128,
                 conv_2_in_dim = 128,
                 conv_2_out_dim = 128,
                 conv_2_kernel_size = 8,
                 conv_2_stride = 4,
                 conv_2_padding = 2,
                 num_norm_features_2 = 128,
                 conv_3_in_dim = 128,
                 conv_3_out_dim = 128,
                 conv_3_kernel_size = 3,
                 conv_3_stride = 2,
                 conv_3_padding = 1,
                 num_norm_features_3 = 128,
                 conv_4_in_dim = 128,
                 conv_4_out_dim = 128,
                 conv_4_kernel_size = 3,
                 conv_4_stride = 1,
                 conv_4_padding = 1,
                 num_norm_features_4 = 128,
                 pooling_kernel_size = 3,
                 pooling_zero_padding = 0,
                 normalization_type = 'layernorm',
                 non_linearity_function = 'gelu',
                 dropout = 0.2):

        super().__init__()
        
        # Batch normalization normalizes each feature separately across all batch samples
        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm1d
        
        # Layer normalization normalizes each each batch sample separately across all features
        elif normalization_type == 'layernorm':
            normalization_layer = LayerNorm
            
        elif normalization_type == None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')
        
        self.conv_layer_1 = Conv1d(in_channels=conv_1_in_dim, out_channels=conv_1_out_dim, kernel_size=conv_1_kernel_size,
                                   stride=conv_1_stride, padding=conv_1_padding)
        self.normalization_1 = normalization_layer(num_norm_features_1)
        
        self.conv_layer_2 = Conv1d(in_channels=conv_2_in_dim, out_channels=conv_2_out_dim, kernel_size=conv_2_kernel_size,
                                   stride=conv_2_stride, padding=conv_2_padding)
        self.normalization_2 = normalization_layer(num_norm_features_2)
        
        self.conv_layer_3 = Conv1d(in_channels=conv_3_in_dim, out_channels=conv_3_out_dim, kernel_size=conv_3_kernel_size,
                                   stride=conv_3_stride, padding=conv_3_padding)
        self.normalization_3 = normalization_layer(num_norm_features_3)
        
        self.conv_layer_4 = Conv1d(in_channels=conv_4_in_dim, out_channels=conv_4_out_dim, kernel_size=conv_4_kernel_size,
                                   stride=conv_4_stride, padding=conv_4_padding)
        self.normalization_4 = normalization_layer(num_norm_features_4)
        
        self.pooling = AvgPool1d(kernel_size=pooling_kernel_size, padding=pooling_zero_padding)
        
        if non_linearity_function == 'relu':
            self.non_linearity = ReLU()
        elif non_linearity_function == 'elu':
            self.non_linearity = ELU()
        elif non_linearity_function == 'gelu':
            self.non_linearity = GELU()
        else:
            sys.exit(f'Wrong value for argument "non_linearity_function": {non_linearity_function}')
        
        self.dropout = Dropout(dropout)
        self.normalization_type = normalization_type


    def forward(self, X):
        
        # We perform random sample dropout as our data augmentation method
        X = self.dropout(X)
        
        # X is now of size [batch_size, frame_len], and we convert it to size [batch_size, 1, frame_len]
        X = X.unsqueeze(1)
        
        if self.normalization_type == 'layernorm':
            X = self.dropout(self.non_linearity(self.normalization_1(self.conv_layer_1(X).permute(0, 2, 1)).permute(0, 2, 1)))
            X = self.dropout(self.non_linearity(self.normalization_2(self.conv_layer_2(X).permute(0, 2, 1)).permute(0, 2, 1)))
            X = self.dropout(self.non_linearity(self.normalization_3(self.conv_layer_3(X).permute(0, 2, 1)).permute(0, 2, 1)))
            X = self.dropout(self.pooling(self.non_linearity(self.normalization_4(self.conv_layer_4(X).permute(0, 2, 1)).permute(0, 2, 1))))
        else:
            X = self.dropout(self.non_linearity(self.normalization_1(self.conv_layer_1(X))))
            X = self.dropout(self.non_linearity(self.normalization_2(self.conv_layer_2(X))))
            X = self.dropout(self.non_linearity(self.normalization_3(self.conv_layer_3(X))))
            X = self.dropout(self.pooling(self.non_linearity(self.normalization_4(self.conv_layer_4(X)))))
        
        X = X.squeeze()
        # X is now of size [batch_size, conv_4_out_dim]
        
        return X







class MLP_classifier(Module):
    """
    A five-layer MLP encoder for 2D inputs (e.g. log-mel features).
    
    """
    
    def __init__(self,
                 linear_1_input_dim = 128,
                 linear_1_output_dim = 128,
                 num_norm_features_1 = 128,
                 linear_2_input_dim = 128,
                 linear_2_output_dim = 128,
                 num_norm_features_2 = 128,
                 linear_3_input_dim = 128,
                 linear_3_output_dim = 2,
                 normalization_type = 'layernorm',
                 non_linearity_function = 'elu',
                 dropout = 0.2):

        super().__init__()
        
        # Batch normalization normalizes each feature separately across all batch samples
        if normalization_type == 'batchnorm':
            normalization_layer = BatchNorm1d
        
        # Layer normalization normalizes each each batch sample separately across all features
        elif normalization_type == 'layernorm':
            normalization_layer = LayerNorm
            
        # Instance normalization normalizes each batch sample and each feature individually
        #elif normalization_type == 'instancenorm':
            #normalization_layer = InstanceNorm1d
            
        elif normalization_type == None:
            normalization_layer = Identity
        else:
            sys.exit(f'Wrong value for argument "normalization_type": {normalization_type}')
        
        self.linear_layer_1 = Linear(in_features=linear_1_input_dim,
                              out_features=linear_1_output_dim)
        self.normalization_1 = normalization_layer(num_norm_features_1)
        
        self.linear_layer_2 = Linear(in_features=linear_2_input_dim,
                              out_features=linear_2_output_dim)
        self.normalization_2 = normalization_layer(num_norm_features_2)
                              
        self.linear_layer_3 = Linear(in_features=linear_3_input_dim,
                              out_features=linear_3_output_dim)
        
        if non_linearity_function == 'relu':
            self.non_linearity = ReLU()
        elif non_linearity_function == 'elu':
            self.non_linearity = ELU()
        elif non_linearity_function == 'gelu':
            self.non_linearity = GELU()
        else:
            sys.exit(f'Wrong value for argument "non_linearity_function": {non_linearity_function}')
        
        self.dropout = Dropout(dropout)


    def forward(self, X):
        
        # X is now of size [batch_size, linear_1_input_dim]
        X = self.dropout(self.non_linearity(self.normalization_1(self.linear_layer_1(X))))
        X = self.dropout(self.non_linearity(self.normalization_2(self.linear_layer_2(X))))
        X = self.non_linearity(self.linear_layer_3(X))
        
        # X_output is of size [batch_size, num_frames_output, linear_3_output_dim]
        
        return X




