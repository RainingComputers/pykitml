import os
from urllib import request

import numpy as np

from .. import pklhandler

'''
This module contains helper functions to download and load
the S1 clustering dataset. 
'''

def get():
    '''
    Downloads the s1 clustering dataset from 
    http://cs.joensuu.fi/sipu/datasets/
    and save is as a pkl file `s1.pkl`.

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
    and you have the `s1.pkl` file, you don't need to call this method again.
    '''
    # Url to download the dataset from
    url = 'http://cs.joensuu.fi/sipu/datasets/s1.txt'

    # Download the dataset
    print('Downloading s1.txt...')
    request.urlretrieve(url, 's1.txt')
    print('Download complete.')

    # Parse the data and save it as a pkl file
    data_array = np.loadtxt('s1.txt')
    pklhandler.save(data_array, 's1.pkl')

    # Delete unnecessary files.
    os.remove('s1.txt')
    print('Deleted unnecessary files.')

def load():
    '''
    Loads x, y points from the s1 clustering dataset from saved pickle file `s1.pkl` to 
    numpy array. S1 clustering dataset contains 15 clusters.

    Returns
    -------
    training_data : numpy.array
        5000x2 numpy array containing x, y points. 

    Raises
    ------
        FileNotFoundError
            If `s1.pkl` file does not exist, i.e, if the dataset was not 
            downloaded and saved using the :py:func:`~get` method.  
    '''
    return pklhandler.load('s1.pkl')
