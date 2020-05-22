import os
from urllib import request

import numpy as np
from numpy import genfromtxt

from .. import pklhandler

'''
This module contains helper functions to download and load
the EEG eye state dataset.
'''

def get():
    '''
    Downloads the eye dataset from
    http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    and saves it as a pkl file `eye.pkl`.
    
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
    and you have the `eye.pkl` file, you don't need to call this method again.   
    '''
    # Url to download the dataset from
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff'

    # Download the dataset
    print('Downloading EEG Eye State.arff')
    request.urlretrieve(url, 'Downloading EEG Eye State.arff')

    # Parse data and save it as a pkl file
    data = genfromtxt('Downloading EEG Eye State.arff', delimiter=',', skip_header=19)
    pklhandler.save(data, 'eye.pkl')

    # Delete unnecessary files
    os.remove('Downloading EEG Eye State.arff')
    print('Deleted unnecessary files.')

def load():
    '''
    Loads the ECG eye state dataset from pkl file.

    All data is from one continuous EEG measurement with the Emotiv EEG 
    Neuroheadset. The duration of the measurement was 117 seconds. 
    The eye state was detected via a camera during the EEG measurement 
    and added later manually to the file after analysing the video frames.
    '1' indicates the eye-closed and '0' the eye-open state. 
    All values are in chronological order with the first measured value 
    at the top of the data.

    Returns
    -------
    inputs : numpy.array
        14980x14 numpy array.
    outputs : numpy.array
        Numpy array containing 14980 outputs.
    '''
    data = pklhandler.load('eye.pkl')
    inputs = data[:, :-1]
    outputs = data[:, -1]

    return inputs, outputs