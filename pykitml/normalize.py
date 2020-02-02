import numpy as np

# ===============================================
# = Functions for Normalization/Feature-scaling =
# ===============================================

def get_minmax(array):
    '''
    Returns two row arrays, one array containing minimum values of each column
    and another one with maximum values.

    Parameters
    ----------
    array : numpy.array
        The array to get minimum and maximum values for.

    Returns
    -------
    array_min : numpy.array
        Array containing minimum values of each column.
    array_max : numpy.array
        Array containing maximum values of each column.
    '''
    return np.amin(array, axis=0), np.amax(array, axis=0)

def normalize_minmax(array, array_min, array_max, cols=[]):
    '''
    Normalizes columns of the array to between 0 and 1 using min-max
    normalization.

    Parameters
    ----------
    array : numpy.array
        The array to normalize.
    array_min : numpy.array
        Array containing minimum values of each column.
    array_max : numpy.array
        Array containing maximum values of each column.
    cols : list
        The columns to normalize. If the list is empty (default),
        all columns will be normalized.

    Returns
    -------
    numpy.array
        The normalized array.

    Note
    ----
    You can use :py:func:`~get_minmax` function to get :code:`array_min`
    and :code:`array_max` parameters.
    '''
    normalized_array = array.astype(float)
    all_normalized = (array - array_min) / (array_max - array_min)

    if(len(cols) == 0):
        # Normalize all columns
        normalized_array = all_normalized
    elif(array.ndim == 1):
        # Normalize only specified columns, 1D array
        normalized_array[cols] = all_normalized[cols]
    else:
        # Normalize onlt specified columns, 2D array
        normalized_array[:, cols] = all_normalized[:, cols]

    return normalized_array

def denormalize_minmax(array, array_min, array_max, cols=[]):
    '''
    Denormalizes columns of a min-max normalized array.

    Parameters
    ----------
    array : numpy.array
        The array to denormalize.
    array_min : numpy.array
        Array containing minimum values of each column.
    array_max : numpy.array
        Array containing maximum values of each column.
    cols : list
        The columns to normalize. If the list is empty (default),
        all columns will be denormalized.

    Returns
    -------
    numpy.array
        The denormalized array.

    Note
    ----
    You can use :py:func:`~get_minmax` function to get :code:`array_min`
    and :code:`array_max` parameters.
    '''
    denormalized_array = array.astype(float)
    all_denormalized = (array * (array_max - array_min)) + array_min

    if(len(cols) == 0):
        # Deormalize all columns
        denormalized_array = all_denormalized
    elif(array.ndim == 1):
        # Denormalize only specified columns, 1D array
        denormalized_array[cols] = all_denormalized[cols]
    else:
        # Denormalize onlt specified columns, 2D array
        denormalized_array[:, cols] = all_denormalized[:, cols]

    return denormalized_array


def get_meanstd(array):
    '''
    Returns two row arrays, one array containing mean of each column
    and another one with standard deviation of each column.

    Parameters
    ----------
    array : numpy.array
        The array to get mean and standard deviation values for.

    Returns
    -------
    array_mean : numpy.array
        Array containing mean values of each column.
    array_stddev : numpy.array
        Array containing standard deviation values of each column.
    '''
    return np.mean(array, axis=0), np.std(array, axis=0)

def normalize_mean(array, array_mean, array_stddev, cols=[]):
    '''
    Normalizes columns of the array with mean normalization.

    Parameters
    ----------
    array : numpy.array
        The array to normalize.
    array_mean : numpy.array
        Array containing mean values of each column.
    array_stddev : numpy.array
        Array containing standard deviation values of each column.
    cols : list
        The columns to normalize. If the list is empty (default),
        all columns will be normalized.


    Returns
    -------
    numpy.array
        The normalized array.

    Note
    ----
    You can use :py:func:`~get_meanstd` function to get :code:`array_mean`
    and :code:`array_stddev` parameters.
    '''
    normalized_array = array.astype(float)
    all_normalized = (array-array_mean)/array_stddev

    if(len(cols) == 0):
        # Normalize all columns
        normalized_array = all_normalized
    elif(array.ndim == 1):
        # Normalize only specified columns, 1D array
        normalized_array[cols] = all_normalized[cols]
    else:
        # Normalize only specified columns, 2D array
        normalized_array[:, cols] = all_normalized[:, cols]

    return normalized_array

def denormalize_mean(array, array_mean, array_stddev, cols=[]):
    '''
    Denormalizes a mean normalized array.

    Parameters
    ----------
    array : numpy.array
        The array to denormalize.
    array_mean : numpy.array
        Array containing mean values of each column.
    array_stddev : numpy.array
        Array containing standard deviation values of each column.

    Returns
    -------
    numpy.array
        The denormalized array.

    Note
    ----
    You can use :py:func:`~get_meanstd` function to get :code:`array_mean`
    and :code:`array_stddev` parameters.
    '''
    denormalized_array = array.astype(float)
    all_denormalized = (array*array_stddev) + array_mean

    if(len(cols) == 0):
        # Denormalize all columns
        denormalized_array = all_denormalized
    elif(array.ndim == 1):
        # Denormalize only specified columns, 1D array
        denormalized_array[cols] = all_denormalized[cols]
    else:
        # Denormalize onlt specified columns, 2D array
        denormalized_array[:, cols] = all_denormalized[:, cols]

    return denormalized_array