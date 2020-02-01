import os
from urllib import request

import numpy as np

from .. import pklhandler

'''
This module contains helper functions to download and load
the heart disease dataset. 
'''

def get():
    '''
    Downloads heartdisease dataset from
    https://archive.ics.uci.edu/ml/datasets/Heart+Disease
    and saves it as a pkl file `heartdisease.pkl`.

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
    and you have the `heartdisease.pkl` file, you don't need to call this method again.
    '''
    # Url to download the dataset from
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'

    # Download the dataset
    print('Downloading processed.cleveland.data...')
    request.urlretrieve(url, 'processed.cleveland.data')
    print('Download complete.')

    # Parse data and save it as a pkl file.
    data_array = []
    # Open the file and put the values in a list.
    with open('processed.cleveland.data', 'r') as datafile:
        for line in datafile:
            try:
                data_array.append(list(map(float, line.split(','))))
            except ValueError:
                continue
    # Convert the list into a numpy array.
    heartdisease_data_array = np.array(data_array)
    # Save as a pkl file.
    pklhandler.save(heartdisease_data_array, 'heartdisease.pkl')

    # Delete unnecessary files.
    os.remove('processed.cleveland.data')
    print('Deleted unnecessary files.')

def load():
    '''
    Loads heart disease dataset from saved pickle file `heartdisease.pkl` to numpy arrays.
    Loads data without any preprocessing.

    Returns
    -------
    inputs : numpy.array
        297x13 numpy array. 297 training examples, each example having 13 inputs(columns).
        The 13 columns corresponds to: 
        :code:`age sex cp trestbps chol fbs restecg thalach exang oldpeak slope ca thal`.

        - age : Age in years
        - sex : 1=male, 0=female
        - cp : Chest pain type (1=typical-angina, 2=atypical-angina 3=non-anginal 4=asymptomatic)
        - trestbps :  Resting blood pressure in mmHg
        - chol : Serum cholesterol in mg/dl
        - fbs : Fasting blood sugar > 120 mg/dl? (1=true, 0=false)
        - restecg : Resting electrocardiographic results (0=normal, 1=ST-T-abnormality 2= left-ventricular-hypertrophy)
        - thalach : Maximum heart rate achieved 
        - exang : Exercise induced angina (1=yes, 0=no) 
        - oldpeak : ST depression induced by exercise relative to rest
        - slope: Slope of the peak exercise ST segment (1=upsloping 2=flat 3=downsloping)
        - ca : Number of major vessels colored by flourosopy (0-3)
        - thal: 3=normal, 6=fixed-defect, 7=reversable-defect

    outputs : numpy.array
        Numpy array with 297 elements. 
        
        - 0: < 50% diameter narrowing 
        - 1: > 50% diameter narrowing    

    Raises
    ------
        FileNotFoundError
            If `heartdisease.pkl` file does not exist, i.e, if the dataset was not 
            downloaded and saved using the :py:func:`~get` method.  
    '''
    # Load data from pkl file.
    heartdisease_data_array = pklhandler.load('heartdisease.pkl')   
    inputs =  heartdisease_data_array[:,:-1]
    outputs = (heartdisease_data_array[:,-1]>0)*1

    # return data
    return inputs, outputs

