import os
from urllib import request

import numpy as np

from .. import pklhandler

'''
This module contains helper functions to download the sonar dataset.
'''

def get():
    '''
    Downloads sonar dataset from
    https://archive.ics.uci.edu/ml/datasets/Connectionist+Bench+(Sonar,+Mines+vs.+Rocks)
    and saves it as a pkl file `sonar.pkl`.

    Raises
    ------
        urllib.error.URLError
            If internet connection is not available or the URL is not accessible.
        OSError
            If the files cannot be created due to a system-related error.
        KeyError
            If invalid/unknown type.

    Note
    ----
    You only need to call this method once, i.e, after the dataset has been downloaded
    and you have the `sonar.pkl` file, you don't need to call 
    this method again.
    '''
    # Url to download the dataset from
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'

    # Download the dataset
    print('Downloading sonar.all-data...')
    request.urlretrieve(url, 'sonar.all-data')
    print('Download complete.')

    out_dict = {
        'R\n':0, 'M\n':1
    }

    # Parse data and save it as pkl file
    data_array = []
    # Open the file and put the values in a list.
    with open('sonar.all-data', 'r') as datafile:
        for line in datafile:
            values = line.split(',')
            values[-1] = out_dict[values[-1]]
            data_array.append(list(map(float, values)))
    # Convert the list to numpy array
    sonar_data_array = np.array(data_array)
    # Save it as a pkl file
    pklhandler.save(sonar_data_array, 'sonar.pkl')

    # Delete files
    os.remove('sonar.all-data')

def load():
    '''
    Loads the adult dataset from `sonar.pkl` file.

    Each pattern is a set of 60 numbers in the range 0.0 to 1.0. 
    Each number represents the energy within a particular frequency band, 
    integrated over a certain period of time. The integration aperture for 
    higher frequencies occur later in time, since these frequencies are 
    transmitted later during the chirp.

    The label associated with each record contains the letter 
    "R" if the object is a rock and "M" if it is a mine (metal cylinder).

    Returns
    -------
    inputs_train : numpy.array
        190x60 numpy array containing training inputs.
    outputs_train : numpy.array
        Numpy array of size 190.
    inputs_test : numpy.array
        18x60 numpy array containing testing inputs.
    outputs_test : numpy.array
        Numpy array of size 18.

    Raises
    ------
        filesNotFoundError
            If `sonar.pkl` file does not exist, 
            i.e, if the dataset was not downloaded and saved using the 
            :py:func:`~get` method.     

    '''
    # Load the data from pkl file
    sonar_data_array = pklhandler.load('sonar.pkl')

    # Split into train and test
    train_neg = sonar_data_array[0:90]
    train_pos = sonar_data_array[97:197]
    test_neg = sonar_data_array[90:97]
    test_pos = sonar_data_array[197:208]

    # Shuffle the dataset, join neg and pos examples
    train = np.concatenate((train_pos, train_neg), axis=0)
    np.random.shuffle(train)
    test = np.concatenate((test_pos, test_neg), axis=0)
    np.random.shuffle(test)

    # Split the dataset into inputs and outputs
    inputs_train = train[:, :-1]
    outputs_train = train[:, -1]
    inputs_test = test[:, :-1]
    outputs_test = test[:, -1]

    # return
    return inputs_train, outputs_train, inputs_test, outputs_test
