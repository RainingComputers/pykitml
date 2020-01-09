import numpy as np

'''
This module contains helper functions to load the fish length dataset.
'''

inputs = np.array([
    # Age Temperature
    [ 14,  25],
    [ 28,  25],
    [ 41,  25],
    [ 55,  25],
    [ 69,  25],
    [ 83,  25],
    [ 97,  25],
    [111,  25],
    [125,  25],
    [139,  25],
    [153,  25],
    [ 14,  27],
    [ 28,  27],
    [ 41,  27],
    [ 55,  27],
    [ 69,  27],
    [ 83,  27],
    [ 97,  27],
    [111,  27],
    [125,  27],
    [139,  27],
    [153,  27],
    [ 14,  29],
    [ 28,  29],
    [ 41,  29],
    [ 55,  29],
    [ 69,  29],
    [ 83,  29],
    [ 97,  29],
    [111,  29],
    [125,  29],
    [139,  29],
    [153,  29],
    [ 14,  31],
    [ 28,  31],
    [ 41,  31],
    [ 55,  31],
    [ 69,  31],
    [ 83,  31],
    [ 97,  31],
    [111,  31],
    [125,  31],
    [139,  31],
    [153,  31]
])

outputs = np.array([
    # Fish-length
     620,
    1315,
    2120,
    2600,
    3110,
    3535,
    3935,
    4465,
    4530,
    4570,
    4600,
     625,
    1215,
    2110,
    2805,
    3255,
    4015,
    4315,
    4495,
    4535,
    4600,
    4600,
     590,
    1305,
    2140,
    2890,
    3920,
    3920,
    4515,
    4520,
    4525,
    4565,
    4566,
     590,
    1205,
    1915,
    2140,
    2710,
    3020,
    3030,
    3040,
    3180,
    3257,
    3214,
])

def load():
    '''
    Loads the fish length dataset without any preprocessing.
    Source: https://people.sc.fsu.edu/~jburkardt/datasets/regression/x06.txt

    The length of a species of fish is to be represented as a function
    of the age and water temperature. The fish are kept in tanks
    at 25, 27, 29 and 31 degrees Celsius.  After birth, a test specimen
    is chosen at random every 14 days and its length measured.

    Returns
    -------
    inputs : numpy.array
        44x2 numpy array, each row having 2 features,
        :code:`age temperature`
    outputs : numpy.array
        Length of fish, numpy array with 44 elements.
    '''
    return inputs, outputs