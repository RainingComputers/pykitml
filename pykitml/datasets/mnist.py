'''
This module contains helper functions to download and load MNIST and MNIST like datasets.   
'''

# ============================================================  
# = Forked from: https://github.com/hsjeong5/MNIST-for-Numpy =
# = Modified with minor changes                              =
# ============================================================

import gzip
import pickle
import os
from urllib import request

import numpy as np

from .. import pklhandler

def get(type = 'classic'):
    '''
    Downloads the MNIST dataset and saves it as a pickle file, `mnist.pkl`.

    Parameters
    ----------
    type : str
        The type of MNIST dataset to download.

        - 'classic' : Downloads the classic hanwritten digits dataset from http://yann.lecun.com/exdb/mnist/
        - 'fashion' : Downloads fashion MNIST from https://github.com/zalandoresearch/fashion-mnist


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
    You only need to call this method once, i.e, after the dataset has been 
    downloaded and you have the `mnist.pkl` file, you don't need to call this method again.
    '''
    # dict of URLs containing MNIST like datasets
    type_URLs = {'classic':'http://yann.lecun.com/exdb/mnist/',
            'fashion':'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    }

    # MNIST files to download
    filename = [
        ['training_images','train-images-idx3-ubyte.gz'],
        ['test_images','t10k-images-idx3-ubyte.gz'],
        ['training_labels','train-labels-idx1-ubyte.gz'],
        ['test_labels','t10k-labels-idx1-ubyte.gz']
    ]

    def download_mnist():
        # Download .gz files
        base_url = type_URLs[type]
        for name in filename:
            print('Downloading '+name[1]+'...')
            request.urlretrieve(base_url+name[1], name[1])
        print('Download complete.')

    def save_mnist():
        # Read .gz files and put them in a numpy array and save it as a pkl file
        mnist = {}
        for name in filename[:2]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
        for name in filename[-2:]:
            with gzip.open(name[1], 'rb') as f:
                mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
        pklhandler.save(mnist, 'mnist.pkl')
        print('Save complete.')

    def clean():
        # Remove unnecessary files
        os.remove('train-images-idx3-ubyte.gz')
        os.remove('t10k-images-idx3-ubyte.gz')
        os.remove('train-labels-idx1-ubyte.gz')
        os.remove('t10k-labels-idx1-ubyte.gz')
        print('Deleted unnecessary files.')

    download_mnist()
    save_mnist()
    clean()


def load():
    '''
    Loads MNIST dataset from saved pickle file `mnist.pkl` to numpy arrays.

    Returns
    -------
        training_data : numpy.array
            60,000x784 numpy array, each row contains flattened version of training images.
        training_targets : numpy.array
            60,000x10 numpy array that contains one hot target array of the corresponding 
            training images.
        testing_data : numpy.array
            10,000x784 numpy array, each row contains flattened version of test images.
        testing_targets : numpy.array
            10,000x10 numpy array that contains one hot target array of the corresponding
            test images.

    Raises
    ------
        FileNotFoundError
            If `mnist.pkl` file does not exist, i.e, if the dataset was not downloaded and
            saved using the :py:func:`~get` method. 
    '''
    mnist = pklhandler.load('mnist.pkl')
    # Normalize data
    training_data = mnist['training_images']/255
    testing_data = mnist['test_images']/255
    # Create one-hot target array for training labels
    training_targets = np.zeros((60000, 10))
    training_targets[np.arange(60000), mnist['training_labels']] = 1
    # Create one-hot target array for testing labels
    testing_targets = np.zeros((10000, 10))
    testing_targets[np.arange(10000), mnist['test_labels']] = 1
    # return the data
    return training_data, training_targets, testing_data, testing_targets


if __name__ == '__main__':
    get()
    