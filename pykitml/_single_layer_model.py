from abc import ABC, abstractmethod

import numpy as np

from ._minimize_model import MinimizeModel

class SingleLayerModel(MinimizeModel, ABC):
    '''
    General base class for single layer models.
    '''

    def __init__(self, input_size, output_size, reg_param=0):
        '''
        Parameters
        ----------
        input_size : int
            Size of input data or number of input features.
        output_size: int
            Number of categories or groups.
        reg_param : int
            Regularization parameter for the model, also known as 'weight decay'.
        '''        
        # Save sizes
        self._input_size = input_size
        self._output_size = output_size

        # Initialize regularization parameter
        self._reg_param = reg_param
        self._reg_param_half = reg_param/2
    
        # Initialize weights and parameters
        epsilon = np.sqrt(6)/(np.sqrt(output_size) + np.sqrt(input_size))
        weights = np.random.rand(output_size, input_size)*2*epsilon - epsilon
        biases = np.random.rand(output_size) * 2 * epsilon - epsilon
        
        # Numpy array to store activations
        self._input_activations = np.array([])
        self._activations = np.array([])
        self._weighted_sum = np.array([])

        # Put parameters in numpy dtype=object array
        W = 0 # Weights
        B = 1 # Biases
        self._params = np.array([None, None], dtype=object)
        self._params[W] = weights
        self._params[B] = biases

    @property
    def _mparams(self):
        return self._params
    
    @_mparams.setter
    def _mparams(self, mparams):
        self._params = mparams

    @property
    def _cost_function(self):
        return self._cost_func

    @property
    def _out_size(self):
        return self._output_size

    def feed(self, input_data):
        # Constants
        W = 0 # Weights
        B = 1 # Biases

        # feed
        self._input_activations = input_data
        self._weighted_sum = (input_data @ self._params[W].T) + self._params[B]
        self._activations = self._activ_func(self._weighted_sum)

    def get_output(self):
        return self._activations.squeeze()

    def _backpropagate(self, index, target):
        # Constants
        W = 0 # Weights
        B = 1 # Biases

        # Gradients
        da_dz = self._activ_func_prime(self._weighted_sum[index], self._activations[index])
        dc_db = self._cost_func_prime(self._activations[index], target) * da_dz
        dc_dw = np.multiply.outer(dc_db, self._input_activations[index])
        
        # Add regularization
        dc_dw += self._reg_param*self._params[W]
        
        # Return gradient
        gradient = np.array([None, None], dtype=object)
        gradient[W] = dc_dw
        gradient[B] = dc_db
        return gradient

    def _get_norm_weights(self):
        W = 0
        return self._reg_param_half*(self._params[W]**2).sum()

    @property
    @abstractmethod
    def _activ_func(self):
        pass

    @property
    @abstractmethod
    def _activ_func_prime(self):
        pass

    @property
    @abstractmethod
    def _cost_func(self):
        pass

    @property
    @abstractmethod
    def _cost_func_prime(self):
        pass