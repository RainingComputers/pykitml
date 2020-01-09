import pickle

'''
This module contains functions for saving and 
loading .pkl files
'''

def save(object_, file_name):
    '''
    Saves an object into a file.

    Parameters
    ----------
    object_ : object
        The object to save
    file_name : str
        The name of the file to save the object in.

    Raises
    ------
        OSError
            If the file cannot be created due to a system-related error.
    '''
    file = open(file_name, 'wb')
    pickle.dump(object_, file)
    file.close()


def load(file_name):    
    '''
    Loads an object from file.

    Parameters
    ----------
    file_name : str
        The name of the file to load the object from.

    Returns
    -------
    object
        The python object stored in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    '''
    file = open(file_name, 'rb')
    object_ = pickle.load(file)
    file.close()
    return object_
