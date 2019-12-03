import numpy as np

# ==============================================
# = Functions for Normalization/Feture-scaling =
# ==============================================

def get_minmax(array):
    '''
    Returns two row vectors, one vector containing minimum values of each column
    and another one with maximum values.

    Parameters
    ----------
    array : numpy.array
        The array to get minimum and maximum values for.

    Returns
    -------
    array_min : numpy.array
        Row vector containing minimum values of each column.
    array_max : numpy.array
        Row vector containing maximum values of each column.
    '''
    return np.amin(array, axis=0), np.amax(array, axis=0)

def normalize_array(array, array_min, array_max):
    '''
    Normalizes every column of the array between 0 and 1 using min-max
    normalization.

    Parameters
    ----------
    array : numpy.array
        The array to normalize.
    array_min : numpy.array
        Row vector containing minimum values of each column.
    array_max : numpy.array
        Row vector containing maximum values of each column.

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

def denormalize_array(output_array, array_min, array_max):
    '''
    Denormalizes a min-max normalized array.

    Parameters
    ----------
    array : numpy.array
        The array to denormalize.
    array_min : numpy.array
        Row vector containing minimum values of each column.
    array_max : numpy.array
        Row vector containing maximum values of each column.

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
    