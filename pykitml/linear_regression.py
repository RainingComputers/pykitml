import numpy as np

from . import _single_layer_model
from . import _functions

class LinearRegression(_single_layer_model.SingleLayerModel):
    '''
    Implements linear regression.
    '''

    @property
    def _activ_func(self):
        return _functions.identity

    @property
    def _activ_func_prime(self):
        return _functions.identity_prime

    @property
    def _cost_func(self):
        return _functions.mse

    @property
    def _cost_func_prime(self):
        return _functions.mse_prime