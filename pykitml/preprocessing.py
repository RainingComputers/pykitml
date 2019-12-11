import numpy as np

'''
    This module contains helper functions for preprocessing data.
'''

# Helper funtion for converting input_array to onehot array
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

def onehot_col(dataset, col):
    '''
    Converts/replaces column of dataset to one-hot values.

    Parameters
    ----------
    dataset : numpy.array
        The input dataset.
    col : int
        The column which has to be replaced/converted
        to one-hot values.

    Returns
    -------
    dataset_new : numpy.array
        The new dataset with replaced column.

    Example
    -------
        >>> import pykitml as pk
        >>> import numpy as np
        >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> pk.onehot_col(a, 0)
        array([[0, 1, 0, 0, 0, 0, 0, 0, 2, 3],
               [0, 0, 0, 0, 1, 0, 0, 0, 5, 6],
               [0, 0, 0, 0, 0, 0, 0, 1, 8, 9]])

    '''
    onehot_col = onehot(dataset[:, col])
    dataset_new = np.delete(dataset, col, axis=1)
    dataset_new = np.insert(dataset_new, [col], onehot_col, axis=1)
    return dataset_new