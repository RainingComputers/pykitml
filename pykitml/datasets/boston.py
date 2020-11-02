import os
from urllib import request

import numpy as np
from numpy import genfromtxt

from .. import pklhandler

'''
This module contains helper functions to download and load
the boston housing dataset.
'''

def get():
    '''
    Downloads the boston dataset from
    https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
    and saves it as a pkl file `boston.pkl`.
    
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
    and you have the `boston.pkl` file, you don't need to call this method again.   
    '''
    # Url to download the dataset from
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
    
    # Download the dataset
    print('Downloading housing.data...')
    request.urlretrieve(url, 'housing.data')
    print('Download complete.')

    # Parse the data and save it as a pkl file
    pklhandler.save(genfromtxt('housing.data'), 'boston.pkl')
    # Delete unnecessary files
    os.remove('housing.data')
    print('Deleted unnecessary files.')

def load():
    '''
    Loads the boston housing dataset from pkl file.

    The inputs have following columns:

    - CRIM :
      per capita crime rate by town
    - ZN :
      proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS :
      proportion of non-retail business acres per town
    - CHAS :    
      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX :      
      nitric oxides concentration (parts per 10 million)
    - RM :       
      average number of rooms per dwelling
    - AGE :      
      proportion of owner-occupied units built prior to 1940
    - DIS :      
      weighted distances to five Boston employment centres
    - RAD :      
      index of accessibility to radial highways
    - TAX :     
      full-value property-tax rate per $10,000
    - PTRATIO : 
      pupil-teacher ratio by town
    - B :       
      1000(Bk - 0.63)^2 where Bk is the proportion of black by town
    - LSTAT :   
      % lower status of the population
    
    The outputs are

    - MEDV :
      Median value of owner-occupied homes in $1000's

    Returns
    -------
    inputs_train : numpy.array
    outputs_train : numpy.array
    inputs_test : numpy.array
    outputs_test : numpy.array

    '''
    data_array = pklhandler.load('boston.pkl')

    inputs_train = data_array[0:500, :-1]
    outputs_train = data_array[0:500, -1]
    inputs_test = data_array[500:, :-1]
    outputs_test = data_array[500:, -1]

    return inputs_train, outputs_train, inputs_test, outputs_test