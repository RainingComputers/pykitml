from copy import deepcopy

import numpy as np
import tqdm

from ._classifier import Classifier
from ._exceptions import _valid_list, InvalidDistributionType

def _gaussian(x, mean, std_dev):
    sqrt_2pi = np.sqrt(2*np.pi)
    return (1/(std_dev*sqrt_2pi))*np.exp(-0.5*(((x-mean)/std_dev)**2))

class NaiveBayes(Classifier):
    '''
    Implements Naive Bayes classifier.

    Note
    ----
    Consider using :class:`.GaussianNaiveBayes` if all of your
    features are continuous.
    '''

    def __init__(self, input_size, output_size, distributions, reg_param=1):
        '''
        Parameters
        ----------
        input_size : int
            Size of input data or number of input features.
        output_size: int
            Number of categories or groups.
        distribution : list
            List of strings describing the distribution to use 
            for each feature. Option are :code:`'gaussian'`, 
            :code:`'binomial'`, :code:`'multinomial'`.
        reg_param : int
            If a given class and feature value never occur together in the training data,
            then the frequency-based probability estimate will be zero.
            This is problematic because it will wipe out all information in the other 
            probabilities when they are multiplied.
            So, the probability will become :code:`log(reg_param)`.
            This is a way to regularize Naive Bayes classifier.
            See https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes

        Raises
        ------
        InvalidDistributionType
            If invalid distribution. Can only be :code:`'gaussian'`, 
            :code:`'binomial'`, :code:`'multinomial'`.
        IndexError
            If the input_size does not match the length of distribution length.
        '''
        # Save info
        self._input_size = input_size
        self._output_size = output_size
        self._reg_param = reg_param

        self._dists = distributions

        # Check if given distributions are valid
        valid_dists = ['gaussian', 'binomial', 'multinomial']
        if not _valid_list(distributions, valid_dists):
            raise InvalidDistributionType

        # indices of categorical features
        self._categorical = [i for i, x in enumerate(distributions) if x!='gaussian']

        # for each class/group, p(class), std_dev, mean, freq and range
        self._pclass = np.zeros((output_size))
        self._mean = np.zeros((output_size, input_size))
        self._std_dev = np.zeros((output_size, input_size))
        self._freqp = [deepcopy([None]*input_size) for x in range(output_size)]
        self._max = None

        # Output
        self._output = None

    @property
    def _out_size(self):
        return self._output_size

    def feed(self, input_data):
        # Make sure array is 2D
        if(input_data.ndim == 1): input_data = np.array([input_data])

        # Set output to correct size
        self._output = np.zeros((input_data.shape[0], self._output_size))
        
        # Loop through each group and find probability
        for C in range(0, self._output_size):
            
            # Loop through each feature and multiply prod(p(xi|Ci))=p(x|Ci)
            # We are doing sum(log(p(xi|Ci))) instead of prod(p(xi|Ci)) to prevent
            # the product from going to zero due to very small numbers.
            # So the result will be log(p(x|Ci)) instead of p(x|Ci)
            p_xci = np.zeros((input_data.shape[0]))
            for x in range(0, self._input_size):
                # p(xi|Ci) if catgorical feature
                if(self._dists[x] != 'gaussian'):
                    p_xici = self._freqp[C][x][input_data[:, x].astype(int)]
                # p(xi|Ci) if continues feature
                else:    
                    p_xici = _gaussian(input_data[:, x], self._mean[C][x], self._std_dev[C][x])
                
                # log(p(x|Ci)) = sum(log(p(xi|Ci)))
                p_xci += np.log(p_xici)
            
            # log(p(Ci|x)) = log(p(Ci))+log(p(x|Ci))
            self._output[:, C] = np.log(self._pclass[C])+p_xci

    def get_output(self):
        return self._output.squeeze()

    def train(self, training_data, targets):
        '''
        Trains the model on the training data.

        Parameters
        ----------
        training_data : numpy.array
            numpy array containing training data.
        targets : numpy.array
            numpy array containing training targets, corresponding to the training data.

        Raises
        ------
        numpy.AxisError
            If output_size is less than two. Use :py:func:`pykitml.onehot` to change
            0/False to [1, 0] and 1/True to [0, 1] for binary classification.

        '''
        print('Training Model...')

        # Get max values for each feature
        self._max = np.amax(training_data, axis=0)

        # Loop through each group
        for group in tqdm.trange(0, self._output_size, ncols=80, unit='groups'):
            # Get examples for the group
            group_examples = self._get_group_examples(training_data, targets, group)
            # Get mean and standard deviation for the group examples
            self._get_mean_std(group, group_examples, training_data.shape[0])
            # Get frequency/probability for categorical 
            self._get_freq(group, group_examples)

    def _get_group_examples(self, training_data, targets, group):
        # Create one hot vector
        gvec = np.zeros((self._output_size))
        gvec[group] = 1

        # Get all example data for this group from training data
        indices = np.argwhere(np.all(targets == gvec, axis=1))
        group_examples = training_data[indices].squeeze()

        # return
        return group_examples

    def _get_mean_std(self, group, group_examples, tot_examples):
        # Get mean, std_dev and p(class)
        self._pclass[group] = group_examples.shape[0]/tot_examples
        self._mean[group] = np.mean(group_examples, axis=0)
        self._std_dev[group] = np.std(group_examples, axis=0)

    def _get_freq(self, group, group_examples):
        # Get frequencies
        for feature in self._categorical:
            # Get frequencies of each category in the feature for this
            # group example
            freqp = np.bincount(group_examples[:, feature].astype(int))
            freqp = freqp/group_examples.shape[0]
            # Replace with reg_param where p(xi|Ci) is zero
            # AKA regularization            
            freqp_reg = np.ones((int(self._max[feature])+1))*self._reg_param
            freqp_reg[0:freqp.shape[0]] = freqp
            self._freqp[group][feature] = freqp_reg


class GaussianNaiveBayes(NaiveBayes):
    def __init__(self, input_size, output_size):
        '''
        Parameters
        ----------
        input_size : int
            Size of input data or number of input features.
        output_size: int
            Number of categories or groups.
        '''
        # Save info
        self._input_size = input_size
        self._output_size = output_size

        # for each class/group, p(class), std_dev, mean, freq and range
        self._pclass = np.zeros((output_size))
        self._mean = np.zeros((output_size, input_size))
        self._std_dev = np.zeros((output_size, input_size))

        # Output
        self._output = None

    def feed(self, input_data):
        # Make sure array in 2D
        if(input_data.ndim == 1): input_data = np.array([input_data])

        # Set output to correct size
        self._output = np.zeros((input_data.shape[0], self._output_size))
        
        # Loop through each group and find probability
        for C in range(0, self._output_size):
            # Calculate p(xi|Ci)
            p_xici = _gaussian(input_data, self._mean[C], self._std_dev[C])
            
            # log(p(x|Ci)) = sum(log(p(xi|Ci)))
            p_xci = np.log(p_xici).sum(axis=1)
            
            # log(p(Ci|x)) = log(p(Ci))+log(p(x|Ci))
            self._output[:, C] = np.log(self._pclass[C])+p_xci

    def train(self, training_data, targets):
        print('Training Model...')

        # Loop through each group
        for group in tqdm.trange(0, self._output_size, ncols=80, unit='groups'):
            # Get examples for the group
            group_examples = self._get_group_examples(training_data, targets, group)
            # Get mean and standard deviation for the group examples
            self._get_mean_std(group, group_examples, training_data.shape[0])
