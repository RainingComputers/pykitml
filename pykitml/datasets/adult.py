import os
from urllib import request

import numpy as np

from .. import pklhandler

'''
This module contains helper functions to download the adult dataset.
'''

def get():
    '''
    Downloads adult dataset from
    https://archive.ics.uci.edu/ml/datasets/adult
    and saves it as a pkl files `adult.data.pkl` and `adult.test.pkl`.

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
    and you have the `adult.data.pkl` and `adult.test.pkl` files, you don't need to call 
    this method again.
    '''
    # Dictionary to store categorical values
    workclass_dict = {
        'Private':0, 'Self-emp-not-inc':1, 'Self-emp-inc':2, 'Federal-gov':3, 
        'Local-gov':4, 'State-gov':5, 'Without-pay':6, 'Never-worked':7
    }

    education_dict = {
        'Bachelors':0, 'Some-college':1, '11th':2, 'HS-grad':3, 'Prof-school':4, 'Assoc-acdm':5,
        'Assoc-voc':6, '9th':7, '7th-8th':8, '12th':9, 'Masters':10, '1st-4th':11, '10th':12,
        'Doctorate':13, '5th-6th':14, 'Preschool':15
    }

    marital_dict = {
        'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2, 'Separated':3, 
        'Widowed':4, 'Married-spouse-absent':5, 'Married-AF-spouse':6
    }

    occupation_dict = {
        'Tech-support':0, 'Craft-repair':1, 'Other-service':2, 'Sales':3, 'Exec-managerial':4, 
        'Prof-specialty':5, 'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8, 
        'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11, 'Protective-serv':12, 
        'Armed-Forces':13
    }

    relationship_dict = {
        'Wife':0, 'Own-child':1, 'Husband':2, 'Not-in-family':3, 'Other-relative':4, 
        'Unmarried':5
    }

    race_dict = {
        'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3, 'Black':4
    }

    sex_dict = {
        'Female':0, 'Male':1
    }

    country_dict = {
        'United-States':0, 'Cambodia':1, 'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5, 
        'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9, 'South':10, 'China':11, 'Cuba':12, 
        'Iran':13, 'Honduras':14, 'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19, 'Mexico':20, 
        'Portugal':21, 'Ireland':22, 'France':23, 'Dominican-Republic':24, 'Laos':25, 'Ecuador':26, 'Taiwan':27, 
        'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31, 'Nicaragua':32, 'Scotland':33, 'Thailand':34, 
        'Yugoslavia':35, 'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39, 'Holand-Netherlands':40,
    }

    out_dict = {
        '<=50K\n':0, '>50K\n':1
    }

    def download(files_name):
        # Url to download the dataset from
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/'+files_name

        # Download the dataset
        print('Downloading ' + files_name + '...')
        request.urlretrieve(url, files_name)
        print('Download complete.')

        # Parse data and save it as a pkl files
        data_array = []
        # Open the files and put the values in a list
        with open(files_name, 'r') as datafiles:
            for line in datafiles:
                try:
                    values = line.replace(' ', '').replace('.', '').split(',')
                    del values[4]
                    values[0] = float(values[0])
                    values[1] = workclass_dict[values[1]]
                    values[2] = float(values[2])
                    values[3] = education_dict[values[3]]
                    values[4] = marital_dict[values[4]]
                    values[5] = occupation_dict[values[5]]
                    values[6] = relationship_dict[values[6]]
                    values[7] = race_dict[values[7]]
                    values[8] = sex_dict[values[8]]
                    values[9] = float(values[9])
                    values[10] = float(values[10])
                    values[11] = float(values[11])
                    values[12] = country_dict[values[12]]
                    values[13] = out_dict[values[13]]
                    data_array.append(values)
                except (KeyError, ValueError, IndexError):
                    continue
        # Convert to numpy array
        np_data = np.array(data_array)

        # save it as a pkl files
        pklhandler.save(np_data, files_name+'.pkl')

    # Download the data set
    download('adult.data')
    download('adult.test')

    # Clean up
    print('Deleting unnecessary files...')
    os.remove('adult.data')
    os.remove('adult.test')


def load():
    '''
    Loads the adult dataset from `adult.data.pkl` and `adult.test.pkl` files.
    The inputs have the following columns:

    - age
    - workclass : 
      Private=0, Self-emp-not-inc=1, Self-emp-inc=2, Federal-gov=3, 
      Local-gov=4, State-gov=5, Without-pay=6, Never-worked=7
    - fnlwgt 
    - education :
      Bachelors=0, Some-college=1, 11th=2, HS-grad=3, Prof-school=4, Assoc-acdm=5,
      Assoc-voc=6, 9th=7, 7th-8th=8, 12th=9, Masters=10, 1st-4th=11, 10th=12,
      Doctorate=13, 5th-6th=14, Preschool=15
    - marital-status :
      Married-civ-spouse=0, Divorced=1, Never-married=2, Separated=3, 
      Widowed=4, Married-spouse-absent=5, Married-AF-spouse=6
    - occupation :
      Tech-support=0, Craft-repair=1, Other-service=2, Sales=3, Exec-managerial=4, 
      Prof-specialty=5, Handlers-cleaners=6, Machine-op-inspct=7, Adm-clerical=8, 
      Farming-fishing=9, Transport-moving=10, Priv-house-serv=11, Protective-serv=12, 
      Armed-Forces=13
    - relationship :
      Wife=0, Own-child=1, Husband=2, Not-in-family=3, Other-relative=4, 
      Unmarried=5
    - race :
      White=0, Asian-Pac-Islander=1, Amer-Indian-Eskimo=2, Other=3, Black=4
    - sex :
      Female=0, Male=1
    - capital-gain
    - capital-loss
    - hours-per-week
    - native-country
      United-States=0, Cambodia=1, England=2, Puerto-Rico=3, Canada=4, Germany=5, 
      Outlying-US(Guam-USVI-etc)=6, India=7, Japan=8, Greece=9, South=10, China=11, Cuba=12, 
      Iran=13, Honduras=14, Philippines=15, Italy=16, Poland=17, Jamaica=18, Vietnam=19, Mexico=20, 
      Portugal=21, Ireland=22, France=23, Dominican-Republic=24, Laos=25, Ecuador=26, Taiwan=27, 
      Haiti=28, Columbia=29, Hungary=30, Guatemala=31, Nicaragua=32, Scotland=33, Thailand=34, 
      Yugoslavia=35, El-Salvador=36, Trinadad&Tobago=37, Peru=38, Hong=39, Holand-Netherlands=40,

    The outputs are:

    - <=50K = 0/False 
    - >50K = 1/True

    Returns
    -------
    inputs_train : numpy.array
        392106x13 numpy array containing training inputs.
    outputs_train : numpy.array
        Numpy array of size 392106.
    inputs_test : numpy.array
        195780x13 numpy array containing testing inputs.
    outputs_test : numpy.array
        Numpy array of size 195780.

    Raises
    ------
        filesNotFoundError
            If `adult.data.pkl` or `adult.test.pkl` files does not exist, 
            i.e, if the dataset was not downloaded and saved using the 
            :py:func:`~get` method.  
    '''
    train = pklhandler.load('adult.data.pkl')
    inputs_train = train[:, :-1]
    outputs_train = train[:, -1]

    test = pklhandler.load('adult.test.pkl')
    inputs_test = test[:, :-1]
    outputs_test = test[:, -1]

    return inputs_train, outputs_train, inputs_test, outputs_test

