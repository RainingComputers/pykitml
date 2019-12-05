import numpy as np

from . import _single_layer_model
from . import _functions

class LogisticRegression(_single_layer_model.SingleLayerModel):
    '''
    Implements logistic regression for classification.
    '''

    @property
    def _activ_func(self):
        return _functions.softmax

    @property
    def _activ_func_prime(self):
        return _functions.softmax_prime

    @property
    def _cost_func(self):
        return _functions.cross_entropy

    @property
    def _cost_func_prime(self):
        return _functions.cross_entropy_prime