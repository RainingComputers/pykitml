import base64
import zlib

import numpy as np

from .. import pklhandler
from ._s1_compressed import encoded_data

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

    # Decompress the data and save it as a pkl file
    decoded_data = base64.decodebytes(encoded_data)
    uncompressed_data = zlib.decompress(decoded_data)
    data_array = np.frombuffer(uncompressed_data, dtype=np.int64).reshape(5000, 2)
    pklhandler.save(data_array, 's1.pkl')


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
