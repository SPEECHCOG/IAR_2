# A PyTorch implementation of the IAR 2.0 algorithm

This repository contains code for refining inconsistent training labels for time-series data using the [iterative annotation refinement (IAR) 2.0 algorithm](https://ieeexplore.ieee.org/document/10854471). The present example uses simulated data, which consists of a binary waveform classification task with three simulated annotators. The code has been implemented using PyTorch. For a thorough description of the IAR 2.0 algorithm, see [the publication](https://ieeexplore.ieee.org/document/10854471).

**The present IAR 2.0 implementation has been used in the following publication:**
[E. Vaaras, M. Airaksinen, and O. Räsänen, "IAR 2.0: An algorithm for refining inconsistent annotations for time-series data using discriminative classifiers", _IEEE Access_, vol. 13, pp. 19979--19995, 2025](https://ieeexplore.ieee.org/document/10854471).

If you use the present code or its derivatives, please cite the [repository URL](https://github.com/SPEECHCOG/IAR_2) and/or the [aforementioned publication](https://ieeexplore.ieee.org/document/10854471).

## Requirements
Any `PyTorch` version newer than version 1.9.0 should work fine. You can find out how to install PyTorch here: https://pytorch.org/get-started/locally/. You also need to have `NumPy`, `scikit-learn`, `Librosa`, and `SciPy` installed.

## Repository contents
- `conf_train_iar_2_model_simulated_data.py`: Example configuration file for running IAR 2.0 using simulated data.
- `iar_2_data_loader.py`: A file containing the the data loader for IAR 2.0 using simulated data.
- `iar_2_model.py`: A file containing the models which were used in the IAR 2.0 simulation experiments.
- `train_iar_2_model_simulated_data.py`: A script for running the IAR 2.0 algorithm for simulated data, and also for testing the classification performance of models trained using the original (discretized) soft labels and the labels produced by the IAR 2.0 algorithm.
- `py_conf_file_into_text.py`: An auxiliary script for converting _.py_ configuration files into lists of text that can be used for printing or writing the configuration file contents into a text file.


## Examples of how to use the code


### How to run IAR 2.0 on simulated data using the standard configuration file:
For running the IAR 2.0 algorithm for simulated data using the configuration file _conf_train_iar_2_model_simulated_data.py_, simply use the command
```
python train_iar_2_model_simulated_data.py
```
and you are good to go. Note that this option requires having a configuration file named _conf_train_iar_2_model_simulated_data.py_ in the same directory as the file _train_iar_2_model_simulated_data.py_.

### How to run IAR 2.0 on simulated data using a custom configuration file:
If you want to use a custom configuration file, use the command
```
python train_iar_2_model_simulated_data.py <configuration_file>
```
to run the IAR 2.0 algorithm for simulated data. Note that _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use during the IAR 2.0 process.
