import numpy as np

from ._single_layer_model import SingleLayerModel
from ._classifier import Classifier
from . import _functions

class LogisticRegression(SingleLayerModel, Classifier):
    '''
    Implements logistic regression for classification.
    '''
    def __init__(self, input_size, output_size, reg_param=0):
        # Initialize base class
        super(LogisticRegression, self).__init__(input_size, output_size, reg_param)

        # Choose output activation function
        if(output_size == 1):
            # For binary classification
            self._afunc = _functions.sigmoid
            self._afunc_prime = _functions.sigmoid_prime
        else:
            # For multiclass classification
            self._afunc = _functions.softmax
            self._afunc_prime = _functions.softmax_prime

    @property
    def _activ_func(self):
        return self._afunc

    @property
    def _activ_func_prime(self):
        return self._afunc_prime

    @property
    def _cost_func(self):
        return _functions.cross_entropy

    @property
    def _cost_func_prime(self):
        return _functions.cross_entropy_prime