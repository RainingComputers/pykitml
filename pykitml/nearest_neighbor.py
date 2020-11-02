import numpy as np

from ._classifier import Classifier
from ._regressor import Regressor
from . import _functions

class NearestNeighbor(Classifier, Regressor):
    '''
    This class implements nearest neighbor classifier.
    '''

    def __init__(self, inputs_size, output_size, no_neighbors=1):
        '''
        Parameters
        ----------
        input_size : int
            Size of input data or number of input features.
        output_size : int
            Number of categories or groups.
        no_neighbors : int
            The number of nearest neighbors to consider.
        '''
        self._k = no_neighbors
        self._output = None

        self._input_size = inputs_size
        self._output_size = output_size

    @property
    def _out_size(self):
        return self._output_size

    def train(self, training_data, targets):
        '''
        Trains the model on the training data.

        Parameters
        ----------
        training_data : numpy.array
            numpy array containing training data.
        targets : numpy.array
            numpy array containing training targets, corresponding to the training data.
        '''
        self._inputs = training_data
        self._outputs = targets

    def feed(self, input_data):
        # Make sure array is 2D
        if(input_data.ndim == 1): input_data = np.array([input_data])

        # Get pair wise distances
        distances = _functions.pdist(input_data, self._inputs)

        # Sort the distances
        indices = np.argsort(distances, axis=1)[:, 0:self._k]

        # Get output
        self._output = np.mean(self._outputs[indices], axis=1)

    def get_output(self):
        return self._output.squeeze()
    
