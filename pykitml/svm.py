import functools

import numpy as np

from ._single_layer_model import SingleLayerModel
from ._classifier import Classifier
from . import _functions

def gaussian_kernel(input_data, training_inputs, sigma=1):
    '''
    Transforms the give input data using the gaussian kernal.

    Parameters
    ----------
    input_data : numpy.array
        The input data points to transform.
    training_inputs : numpy.array
        The training data.
    sigma : float
        Hyperparameter that determines the 'spread' of the kernel.

    '''
    # Calculate squared L2 norm of each data point with 
    # every other data point
    distances = _functions.pdist(input_data, training_inputs)
    # Apply gaussian kernel
    transformed_inputs = np.exp((-1/(2*sigma**2))*distances)
    # return
    return transformed_inputs

class SVM(SingleLayerModel, Classifier):
    '''
    Implements Support Vector Machine with Linear Kernel.

    Note
    ----
    The outputs/targets in the training/testing data should have :code:`-1` instead
    of :code:`0` for training. See example for more details.
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

