from itertools import combinations_with_replacement

import numpy as np

'''
This module contains helper functions for preprocessing data.
'''

def onehot(input_array):
    '''
    Converts input array to one-hot array.

    Parameters
    ----------
    input_array : numpy.array
        The input numpy array.
    
    Returns
    -------
    one_hot : numpy.array
        The converted onehot array.

    Example
    -------
    >>> import numpy as np
    >>> import pykitml as pk
    >>> a = np.array([0, 1, 2])
    >>> pk.onehot(a)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    '''
    array = input_array.astype(int)
    one_hot = np.zeros((array.size, array.max()+1))
    one_hot[np.arange(array.size), array] = 1
    return one_hot

def onehot_cols(dataset, cols):
    '''
    Converts/replaces columns of dataset to one-hot values.

    Parameters
    ----------
    dataset : numpy.array
        The input dataset.
    cols : list
        The columns which has to be replaced/converted
        to one-hot values.

    Returns
    -------
    dataset_new : numpy.array
        The new dataset with replaced columns.

    Example
    -------
        
        >>> import pykitml as pk
        >>> import numpy as np
        >>> a = np.array([[0, 1, 2.2], [1, 2, 3.4], [0, 0, 1.1]]) 
        >>> a
        array([[0. , 1. , 2.2],
               [1. , 2. , 3.4],
               [0. , 0. , 1.1]])
        >>> pk.onehot_cols(a, cols=[0, 1])
        array([[1. , 0. , 0. , 1. , 0. , 2.2],
               [0. , 1. , 0. , 0. , 1. , 3.4],
               [1. , 0. , 1. , 0. , 0. , 1.1]])

    '''
    offset=0
    dataset_new = dataset
    for col in cols:
        onehot_colmn = onehot(dataset_new[:, col+offset])
        dataset_new = np.delete(dataset_new, col+offset, axis=1)
        dataset_new = np.insert(dataset_new, [col+offset], onehot_colmn, axis=1)
        offset += onehot_colmn.shape[1]-1 

    return dataset_new

def onehot_cols_traintest(dataset_train, dataset_test,  cols):
    '''
    Converts/replaces columns of :code:`dataset_train` and 
    :code:`dataset_test` to one-hot values.

    Parameters
    ----------
    dataset_train : numpy.array
        The training dataset.
    dataset_test : numpy.array
        The testing dataset.
    cols : list
        The columns which has to be replaced/converted
        to one-hot values.

    Returns
    -------
    dataset_train_new : numpy.array
        The new training dataset with replaced columns.
    dataset_test_new : numpy.array
        The new testing dataset with replaced columns.

    Example
    -------
        
        >>> import pykitml as pk
        >>> import numpy as np
        >>> a_train = np.array([[0, 1, 3.2], [1, 2, 3.5], [0, 0, 3.4]])
        >>> a_test = np.array([[0, 3, 3.2], [1, 2, 4.5], [1, 3, 4.5]])
        >>> a_train_onehot, a_test_onehot = pk.onehot_cols_traintest(a_train, a_test, cols=[0,1])
        >>> a_train_onehot
        array([[1. , 0. , 0. , 1. , 0. , 0. , 3.2],
               [0. , 1. , 0. , 0. , 1. , 0. , 3.5],
               [1. , 0. , 1. , 0. , 0. , 0. , 3.4]])
        >>> a_test_onehot
        array([[1. , 0. , 0. , 0. , 0. , 1. , 3.2],
               [0. , 1. , 0. , 0. , 1. , 0. , 4.5],
               [0. , 1. , 0. , 0. , 0. , 1. , 4.5]])

    '''
    # Combine the datasets
    dataset_new = np.concatenate((dataset_train, dataset_test), axis=0)
    
    # Replace columns with on hot values
    offset=0
    for col in cols:
        onehot_colmn = onehot(dataset_new[:, col+offset])
        dataset_new = np.delete(dataset_new, col+offset, axis=1)
        dataset_new = np.insert(dataset_new, [col+offset], onehot_colmn, axis=1)
        offset += onehot_colmn.shape[1]-1 

    split = dataset_train.shape[0]
    return dataset_new[:split, :], dataset_new[split:, :]

def polynomial(dataset_inputs, degree=3, cols=[]):
    '''
    Generates polynomial features from the input dataset.
    For example, if an input sample is two dimensional and of the form [a, b], 
    the degree-2 polynomial features are :code:`[a, b, a^2, ab, b^2]`, and degree-3
    polynomial features are 
    :code:`[a, b, a^2, ab, b^2, a^3, (a^2)*b, a*(b^2), b^3]`.

    Parameters
    ----------
    dataset_inputs : numpy.array
        The input dataset to generate the ploynomials from.
    degree : int
        The dgree of the polynomial.
    cols : list
        The columns to use to generate polynomial features, columns
        not in this list will be ignored. If empty (default), all columns will
        used to generate polynomial features.

    Returns
    -------
    numpy.array
        The new dataset with polynomial features.

    Example
    -------

        >>> import numpy as np
        >>> import pykitml as pk
        >>> pk.polynomial(np.array([[1, 2], [2, 3]]), degree=2)
        array([[1., 2., 1., 2., 4.],
               [2., 3., 4., 6., 9.]])
        >>> pk.polynomial(np.array([[1, 2], [2, 3]]), degree=3)
        array([[ 1.,  2.,  1.,  2.,  4.,  1.,  2.,  4.,  8.],
               [ 2.,  3.,  4.,  6.,  9.,  8., 12., 18., 27.]])
        >>> pk.polynomial(np.array([[1, 4, 5, 2], [2, 5, 6, 3]]), degree=2, cols=[0, 3])
        array([[1., 4., 5., 2., 1., 2., 4.],
               [2., 5., 6., 3., 4., 6., 9.]])
               
    '''
    # Make sure 2D array
    if(dataset_inputs.ndim == 1):
        inputs = np.array([dataset_inputs])
    else:
        inputs = dataset_inputs

    # Choose the columns to genrate polynomial features for
    if(len(cols) == 0): cols = range(inputs.shape[1])
    
    poly_dataset = inputs

    # Generate degree terms
    for d in range(2, degree+1):
        # Generate terms indices for degree d
        term_indices = list(combinations_with_replacement(cols, r=d))
        # Multiply them to form the term and concatenate
        for indices in term_indices:
            term = inputs[:, indices].prod(axis=1)
            temp = np.zeros((poly_dataset.shape[0], poly_dataset.shape[1]+1))
            temp[:, :-1] = poly_dataset
            temp[:, -1] = term
            poly_dataset = temp

    return poly_dataset.squeeze() 

