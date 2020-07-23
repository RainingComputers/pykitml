from abc import ABC, abstractmethod
import itertools

import numpy as np
import matplotlib.pyplot as plt

import tqdm

from . import _heatmap
from . import preprocessing

class Regressor(ABC):
    '''
    Mix-in class for Regression models.
    '''

    @abstractmethod
    def get_output(self):
        '''
        Returns the output activations of the model.

        Returns
        -------
        numpy.array
            The output activations.
        '''
        pass

    @abstractmethod
    def feed(self, input_data):
        '''
        Accepts input array and feeds it to the model.

        Parameters
        ----------
        input_data : numpy.array
            The input to feed the model.

        Raises
        ------
        ValueError
            If the input data has invalid dimensions/shape.      

        Note
        ----
        This function only feeds the input data, to get the output after calling this
        function use :py:func:`get_output` or :py:func:`get_output_onehot`
        '''
        pass

    @property
    @abstractmethod
    def _out_size(self):
        '''
        Returns number of nodes/neurons in the output layer.
        '''
        pass

    def r2score(self, testing_data, testing_targets):
        '''
        Return R-squared or coefficient of determination value.
        
        Parameters
        ----------
        testing_data : numpy.array
            numpy array containing testing data.
        testing_targets : numpy.array
            numpy array containing testing targets, corresponding to the testing data.
        
        Returns
        -------
        r2score : float
            The average cost of the model over the testing data.

        Raises
        ------
        ValueError
            If :code:`testing_data` or :code:`testing_tagets` has invalid dimensions/shape.      
        '''
        self.feed(testing_data)
        output = self.get_output()

        error = ((output-testing_targets)**2).sum()
        var = ((testing_targets-testing_targets.mean(axis=0)) ** 2).sum()

        return 1-error/var

