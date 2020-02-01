import os
from urllib import request

import numpy as np
from numpy import genfromtxt

from .. import pklhandler

'''
This module contains helper functions to download and load
the banknote dataset.
'''

def get():
    '''
    Downloads the banknote dataset from
    http://archive.ics.uci.edu/ml/datasets/banknote+authentication
    and saves it as a pkl file `banknote.pkl`.
    
    Raises
    ------
        urllib.error.URLError
            If internet connection is not available or the URL is not accessible.
        OSError
            If the file cannot be created due to a system-related error.
        KeyError
            If invalid/unknown type.

    Note
    ----
    You only need to call this method once, i.e, after the dataset has been downloaded
    and you have the `banknote.pkl` file, you don't need to call this method again.   
    '''
    # Url to download the dataset from
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'
    
    # Download the dataset
    print('Downloading data_banknote_authentication.txt')
    request.urlretrieve(url, 'data_banknote_authentication.txt')
    print('Download complete.')

    # Parse the data and save it as a pkl file
    pklhandler.save(genfromtxt('data_banknote_authentication.txt', delimiter=','), 'banknote.pkl')

    # Delete unnecessary files
    os.remove('data_banknote_authentication.txt')
    print('Deleted unnecessary files.')

def load():
    '''
    Loads the banknote data from pkl file.
    
    The inputs have the following columns:

    - Variance of Wavelet Transformed image (continuous)
    - Skewness of Wavelet Transformed image (continuous) 
    - Curtosis of Wavelet Transformed image (continuous) 
    - Entropy of image (continuous) 

    The outputs are:

    - 0 = Real
    - 1 = Counterfeit

    Returns
    -------
    inputs_train : numpy.array
        1102x4 numpy array containing training inputs.
    outputs_train : numpy.array
        Numpy array of size 1102.
    inputs_test : numpy.array
        270x4 numpy array containing testing inputs.
    outputs_test : numpy.array
        Numpy array of size 270.
    
    '''
    data_array = pklhandler.load('banknote.pkl')

    # Separate data, positive and negative examples
    negative_examples = data_array[:762]
    positive_examples = data_array[762:]

    # Separate into training and testing
    negative_examples_test = negative_examples[:150]
    negative_examples_train = negative_examples[150:]
    positive_examples_test = positive_examples[:120]
    positive_examples_train = positive_examples[120:]

    # Join them to form training and testing dataset
    train = np.concatenate((negative_examples_train, positive_examples_train), axis=0)
    test = np.concatenate((negative_examples_test, positive_examples_test), axis=0)

    # Shuffle the dataset
    shuff_indices = np.arange(train.shape[0])
    np.random.shuffle(shuff_indices)    
    train = train[shuff_indices]
    shuff_indices = np.arange(test.shape[0])
    np.random.shuffle(shuff_indices)   
    test = test[shuff_indices]

    inputs_train = train[:, :-1]
    outputs_train = train[:, -1]
    inputs_test = test[:, :-1]
    outputs_test = test[:, -1]

    return inputs_train, outputs_train, inputs_test, outputs_test
        