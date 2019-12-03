'''
    This module contains helper functions to download and load the diabetes dataset.   
'''   

from urllib import request
import os

import numpy as np

from . import pklhandler

def get():
    '''
    Downloads diabetes dataset from https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html
    and saves it as a pkl file `diabetes.pkl`.

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
    and you have the `diabetes.pkl` file, you don't need to call this method again.
    '''
    def download_diabetes():
        # Download data from URL.
        data_url = 'https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt'
        print('Downloading diabetes.tab.txt...')
        request.urlretrieve(data_url, 'diabetes.tab.txt')
        print('Download complete.')

    def save_diabetes():
        # Parse data and save it as a pkl file.
        data_array = []
        # Open the file and put the values in a list.
        with open('diabetes.tab.txt', 'r') as datafile:
            # Skip first line and parse the rest.
            datafile.readline()
            for line in datafile:
                data_array.append(list(map(float, line.split())))
        # Convert the list into a numpy array.
        diabetes_data_array = np.array(data_array)
        # Save as a pkl file.
        pklhandler.save(diabetes_data_array, 'diabetes.pkl')

    def clean():
        # Delete unecessary files.
        os.remove('diabetes.tab.txt')
        print('Deleted unecessary files.')

    download_diabetes()
    save_diabetes()
    clean()


def load():
    '''
    Loads diabetes dataset from saved pickle file `diabetes.pkl` to numpy arrays.
    The data is not split and is not normalized. Loads data without any preprocessing.

    Returns
    -------
    input : numpy.array
        442x10 numpy array. 442 training examples, each example having 10 inputs(columns).
        The 10 columns correspond to: :code:`AGE SEX BMI BP S1 S2 S3 S4 S5 S6`
    output : numpy.array
        numpy array with 442 elements.
        
    Raises
    ------
        FileNotFoundError
            If `diabetes.pkl` file does not exist, i.e, if the dataset was not 
            downloaded and saved using the :py:func:`~get` method.     
    '''
    # Load data from pkl file.
    diabetes_data_array = pklhandler.load('diabetes.pkl')   
    # return data
    return diabetes_data_array[:,:-1], diabetes_data_array[:,-1]

if __name__ == '__main__':
    get()
    