import numpy as np

'''
This module contains probability distribution functions
'''

def gaussian(x, mean, std_dev):
    sqrt_2pi = np.sqrt(2*np.pi)
    return (1/(std_dev*sqrt_2pi))*np.exp(-0.5*(((x-mean)/std_dev)**2))

def binomial():
    '''
    Will be added in future versions
    '''
    pass

def multinomial():
    '''
    Will be added in future versions
    '''
    pass
    