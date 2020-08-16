import numpy as np

'''
This module contains utility functions
'''

# =====================
# = Utility functions =
# =====================

def pdist(x, y):
    '''
    Calculate pairwise square distances between matrix x and y.
    See: https://stackoverflow.com/a/56084419/5516481
    '''
    if(x.ndim==1): x = np.array([x])

    nx, p = x.shape
    x_ext = np.empty((nx, 3*p))
    x_ext[:, :p] = 1
    x_ext[:, p:2*p] = x
    x_ext[:, 2*p:] = np.square(x)

    ny = y.shape[0]
    y_ext = np.empty((3*p, ny))
    y_ext[:p] = np.square(y).T
    y_ext[p:2*p] = -2*y.T
    y_ext[2*p:] = 1

    return x_ext.dot(y_ext)

# ==============================================
# = Activation functions and their derivatives =
# ==============================================

def sigmoid(weighted_sum):
    '''
    Returns sigmoid of the weighted sum array of a layer.
    '''
    return 1 / (1 + np.exp(-weighted_sum))

def sigmoid_prime(weighted_sum, activations):
    '''
    Returns the derivative of sigmoid w.r.t layer's weighted sum.
    '''
    return activations * (1 - activations)

def tanh(weighted_sum):
    '''
    Returns tanh of the weighted sum array of a layer.
    '''
    return np.tanh(weighted_sum)

def tanh_prime(weighted_sum, activations):
    '''
    Returns the derivative of tanh w.r.t layer's weighted sum.
    '''
    return 1 - (activations ** 2)

def leakyrelu(weighted_sum):
    '''
    Returns leaky-ReLU of the weighted sum array of a layer.
    '''
    return np.where(weighted_sum > 0, weighted_sum, 0.01 * weighted_sum)

def leakyrelu_prime(weighted_sum, activations):
    '''
    Returns the derivative of leaky-ReLU w.r.t layer's weighted sum.
    '''
    return np.where(weighted_sum > 0, 1, 0.01)

def relu(weighted_sum):
    '''
    Returns ReLU of the weighted sum array of a layer.
    '''
    return np.where(weighted_sum > 0, weighted_sum, 0)

def relu_prime(weighted_sum, activations):
    '''
    Returns the derivative of ReLU w.r.t layer's weighted sum.
    '''
    return np.where(weighted_sum > 0, 1, 0)

def softmax(weighted_sum):
    '''
    Returns softmax of the weighted sum array of a layer.
    If weighted_sum is a 2D array, then it performs softmax over each row.
    '''
    if(weighted_sum.ndim == 1):
        exps = np.exp(weighted_sum - np.max(weighted_sum))
        return exps / np.sum(exps)
    else:
        normalized = weighted_sum - np.expand_dims(np.max(weighted_sum, axis=1), axis=1)
        exps = np.exp(normalized)
        return exps / np.expand_dims(np.sum(exps, axis=1), 1)

def identity(weighted_sum):
    '''
    Returns identity of the weighted sum array of a layer.
    '''
    return weighted_sum

def identity_prime(weighted_sum, activations):
    '''
    Returns the derivative of identity w.r.t layer's weighted sum.
    '''
    return 1

def softmax_prime(weighted_sum, activations):
    '''
    Returns the derivative of softmax w.r.t layer's weighted sum.
    '''
    return activations * (1 - activations)

# ========================================
# = Cost functions and their derivatives =
# ========================================

def mse(output, target):
    '''
    Returns mean squared error cost of the output.
    '''
    return 0.5 * ((output - target) ** 2)

def mse_prime(output, target):
    '''
    Returns the derivative of the mse cost.
    '''
    return (output-target)

def cross_entropy(output, target):
    '''
    Returns cross entropy cost of the output.
    '''
    return -(target * np.log(output)) - ((1-target) * np.log(1-output))

def cross_entropy_prime(output, target):
    '''
    Returns the derivative of the cross entropy cost.
    '''
    return (output-target) / (output * (1-output))

def hinge_loss(output, target):
    '''
    Returns hinge loss of the output for SVMs.
    '''
    return np.maximum(0, 1 - target*output)

def hinge_loss_prime(output, target):
    '''
    Returns derivative of hinge loss.
    '''
    return np.where((target*output)>1, 0, -1*target)

def huber(output, target):
    '''
    Returns huber loss for dqn
    '''
    error = output - target
    
    is_small_error = np.abs(error) < 1
    
    squared_loss = np.square(error)/2
    linear_loss = np.abs(error) - 0.5
    
    return np.where(is_small_error, squared_loss, linear_loss)

def huber_prime(output, target):
    '''
    Returns derivative of huber loss.
    '''
    error = output - target
    
    is_small_error = np.abs(error) < 1

    return np.where(is_small_error, error, np.sign(error))