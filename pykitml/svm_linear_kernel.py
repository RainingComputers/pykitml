import numpy as np

from . import _single_layer_model
from . import _minimize_model
from . import _functions

class LinearSVM(_single_layer_model.SingleLayerModel, _minimize_model.Classifier):
    '''
    Implements Support Vector Machine with Linear Kernel.
    '''

    @property
    def _activ_func(self):
        return _functions.identity

    @property
    def _activ_func_prime(self):
        return _functions.identity_prime

    @property
    def _cost_func(self):
        return _functions.hinge_loss

    @property
    def _cost_func_prime(self):
        return _functions.hinge_loss_prime