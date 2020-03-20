import numpy as np

from ._single_layer_model import SingleLayerModel
from ._regressor import Regressor
from . import _functions

class LinearRegression(SingleLayerModel, Regressor):
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