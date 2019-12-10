from copy import deepcopy

import numpy as np
import tqdm

from . import _base
from . import _distributions

class NaiveBayes(_base.Classifier):
    '''
    Implements Naive Bayes classifier.
    '''

    def __init__(self, input_size, output_size, distributions=[], reg_param=0):
        '''
        Parameters
        ----------
        input_size : int
            Size of input data or number of input features.
        output_size: int
            Number of categories or groups.
        distribution : list
            List of strings describing the distribution to use 
            for each feature. Option are :code:`'normal'`, 
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
        AttributeError
            If invalid distribution.
        IndexError
            If the input_size does not match the length of distribution length.
        '''
        # Save info
        self._input_size = input_size
        self._output_size = output_size
        self._reg_param = reg_param

        # Choose distribution
        if(len(distributions) == 0):
            distributions = ['normal']*input_size

        self._dists = distributions

        # Get distribution function
        self._pdist = [getattr(_distributions, dist_name) for dist_name in distributions]

        # indices of categorical features
        self._categorical = [i for i, x in enumerate(distributions) if x!='normal']

        # for each class/group, p(class), std_dev, mean, freq and range
        self._pclass = np.zeros((output_size))
        self._mean = np.zeros((output_size, input_size))
        self._std_dev = np.zeros((output_size, input_size))
        self._freqp = [deepcopy([None]*input_size) for x in range(output_size)]
        self._max = np.zeros((output_size, input_size))

        # Output
        self._output = None

    @property
    def _out_size(self):
        return self._output_size

    def feed(self, input_data):
        # Set output to correct size
        self._output = np.zeros((input_data.shape[0], self._output_size))
        
        # Loop through each group and find probabilty
        for C in range(0, self._output_size):
            # Loop through each feature and multiply prod(p(xi|Ci))=p(x|Ci)
            # We are doing sum(log(p(xi|Ci))) instead of prod(p(xi|Ci)) to prevent
            # the product from going to zero due to very small numbers.
            # So the result will be log(p(x|Ci)) instead of p(x|Ci)
            p_xci = np.zeros((input_data.shape[0]))
            for x in range(0, self._input_size):
                # p(xi|ci) if catgorical feature
                if(self._dists[x] != 'normal'):
                    # Get p(xi|ci)
                    p_xici = self._freqp[C][x][input_data[:, x].astype(int)]
                    continue
                # p(xi|ci) if continues feature
                p_xici = self._pdist[x](input_data[:, x], self._mean[C][x], self._std_dev[C][x])
                
                # log(x|ci) = sum(log(p(xi|ci)))
                p_xci += np.log(p_xici)
            
            # Ci = p(Ci)*p(x|Ci)
            self._output[:, C] = np.log(self._pclass[C])+p_xci

    def get_output(self):
        return self._output

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
        print('Training Model...')

        # Split the training data for each group
        for group in tqdm.trange(0, self._output_size, ncols=80, unit='groups'):
            # Create one hot vector
            gvec = np.zeros((self._output_size))
            gvec[group] = 1

            # Get all example data for this group from training data
            indices = np.argwhere(np.all(targets == gvec, axis=1))
            group_examples = training_data[indices].squeeze()

            # Get mean, std_dev and p(class)
            self._pclass[group] = group_examples.shape[0]/training_data.shape[0]
            self._mean[group] = np.mean(group_examples, axis=0)
            self._std_dev[group] = np.std(group_examples, axis=0)

            # Get the range of each group
            self._max[group] = np.amax(group_examples, axis=0)

            # Get frequencies
            for feature in self._categorical:
                freqp = np.bincount(group_examples[:, feature].astype(int))
                freqp = freqp/group_examples.shape[0]
                self._freqp[group][feature] = freqp

        # Pad with reg_param where p(xi|ci) is zero
        # AKA regulerization
        print('Padding categorical features...')
        self._max = np.amax(self._max, axis=0)
        for feature in tqdm.tqdm(self._categorical, ncols=80, unit='features'):
            for group in range(0, self._output_size):
                freqp = self._freqp[group][feature]
                pad_length = int(self._max[feature]-freqp.shape[0])+1
                pad_freqp = np.pad(freqp, (0, pad_length), 'constant', constant_values=0)
                pad_freqp = np.where(pad_freqp==0, self._reg_param, pad_freqp)
                self._freqp[group][feature] = pad_freqp