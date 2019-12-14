import numpy as np

# ======================================
# = Probability Distribution functions =
# ======================================

def gaussian(x, mean, std_dev):
    sqrt_2pi = np.sqrt(2*np.pi)
    return (1/(std_dev*sqrt_2pi))*np.exp(-0.5*(((x-mean)/std_dev)**2))

def binomial():
    pass

def multinomial():
    pass
    