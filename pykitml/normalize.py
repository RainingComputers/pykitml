import numpy as np

# ==============================================
# = Functions for Normalization/Feture-scaling =
# ==============================================

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

def normalize_minmax(array, array_min, array_max):
    '''
    Normalizes every column of the array between 0 and 1 using min-max
    normalization.

    Parameters
    ----------
    array : numpy.array
        The array to normalize.
    array_min : numpy.array
        Array containing minimum values of each column.
    array_max : numpy.array
        Array containing maximum values of each column.

    Returns
    -------
    numpy.array
        The normalized array.

    Note
    ----
    You can use :py:func:`~get_minmax` function to get :code:`array_min`
    and :code:`array_max` parameters.
    '''
    return (array - array_min) / (array_max - array_min)

def denormalize_minmax(output_array, array_min, array_max):
    '''
    Denormalizes a min-max normalized array.

    Parameters
    ----------
    array : numpy.array
        The array to denormalize.
    array_min : numpy.array
        Array containing minimum values of each column.
    array_max : numpy.array
        Array containing maximum values of each column.

    Returns
    -------
    numpy.array
        The denormalized array.

    Note
    ----
    You can use :py:func:`~get_minmax` function to get :code:`array_min`
    and :code:`array_max` parameters.
    '''
    return (output_array * (array_max - array_min)) + array_min

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

def normalize_mean(array, array_mean, array_stddev):
    '''
    Normalizes every column of the array mean normalization.

    Parameters
    ----------
    array : numpy.array
        The array to normalize.
    array_mean : numpy.array
        Array containing mean values of each column.
    array_stddev : numpy.array
        Array containing standard deviation values of each column.

    Returns
    -------
    numpy.array
        The normalized array.

    Note
    ----
    You can use :py:func:`~get_meanstd` function to get :code:`array_mean`
    and :code:`array_stddev` parameters.
    '''
    return (array-array_mean)/array_stddev

def denormalize_mean(array, array_mean, array_stddev):
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
    return (array*array_stddev)+array_mean