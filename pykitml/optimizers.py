from abc import ABC, abstractmethod

import numpy as np

'''
Optimizers module,
REF: http://cs231n.github.io/neural-networks-3/
'''

class Optimizer(ABC):
    '''
    This class is the base class for all optimizers
    '''
    @abstractmethod
    def _optimize(self, parameter, parameter_gradient):
        pass

    @property
    @abstractmethod
    def _mlearning_rate(self):
        pass

    @_mlearning_rate.setter
    @abstractmethod
    def _mlearning_rate(self, learning_rate):
        pass

    @property
    @abstractmethod
    def _mdecay_rate(self):
        pass

    def _decay(self):
        self._mlearning_rate = self._mdecay_rate*self._mlearning_rate


class GradientDescent(Optimizer):
    '''
    This class implements gradient descent optimization.
    '''
    def __init__(self, learning_rate, decay_rate=1):
        '''
        Parameters
        ----------
            learning_rate : float
            decay_rate : float
                Decay rate for leraning rate
        '''
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate

    @property
    def _mlearning_rate(self):
        return self._learning_rate

    @_mlearning_rate.setter
    def _mlearning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def _mdecay_rate(self):
        return self._decay_rate

    def _optimize(self, parameter, parameter_gradient):
        # Update and return the parameter
        return parameter - (self._learning_rate * parameter_gradient)


class Momentum(Optimizer):
    '''
    This class implements momentum optimization.
    '''
    def __init__(self, learning_rate, decay_rate=1, beta=0.9):
        '''
        Parameters
        ----------
            learning_rate : float
            decay_rate : float
                Decay rate for leraning rate
            beta : float
                Should be between 0 to 1.
        '''
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._beta = beta
        self._v = 0

    @property
    def _mlearning_rate(self):
        return self._learning_rate

    @_mlearning_rate.setter
    def _mlearning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def _mdecay_rate(self):
        return self._decay_rate

    def _optimize(self, parameter, parameter_gradient):
        # Integrate v
        self._v = (self._beta*self._v) - (self._learning_rate*parameter_gradient)
        # Update and return the parameter
        return parameter + self._v


class Nesterov(Optimizer):
    '''
    This class implements neterov momentum optimization.
    '''
    def __init__(self, learning_rate, decay_rate=1, beta=0.9):
        '''
        Parameters
        ----------
            learning_rate : float
            decay_rate : float
                Decay rate for leraning rate
            beta : float
                Should be between 0 to 1.
        '''
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._beta = beta
        self._v = 0
        self._v_prev = 0

    @property
    def _mlearning_rate(self):
        return self._learning_rate

    @_mlearning_rate.setter
    def _mlearning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def _mdecay_rate(self):
        return self._decay_rate

    def _optimize(self, parameter, parameter_gradient):
        # Back up before updating
        self._v_prev = self._v
        # Integrate v
        self._v = (self._beta*self._v) - (self._learning_rate*parameter_gradient)
        # Update and return the parameter
        return parameter - (self._beta*self._v_prev) + ((1+self._beta)*self._v)


class Adagrad(Optimizer):
    '''
    This class implements adagrad optmization.
    '''
    def __init__(self, learning_rate, decay_rate=1):
        '''
        Parameters
        ----------
            learning_rate : float
            decay_rate : float
                Decay rate for leraning rate
        '''
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._cache = 0

    @property
    def _mlearning_rate(self):
        return self._learning_rate

    @_mlearning_rate.setter
    def _mlearning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def _mdecay_rate(self):
        return self._decay_rate

    def _optimize(self, parameter, parameter_gradient):
        # For numerical stability
        eps = 10e-8
        # Calculate cache
        self._cache += parameter_gradient**2
        # Update parameter and return
        return parameter + (-self._learning_rate*parameter_gradient)/((self._cache**0.5)+eps)


class RMSprop(Optimizer):
    '''
    This class implements RMSprop optimization.
    '''
    def __init__(self, learning_rate, decay_rate=1, beta=0.9):
        '''
        Parameters
        ----------
            learning_rate : float
            decay_rate : float
                Decay rate for leraning rate
            beta : float
                Should be between 0 to 1.
        '''
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._beta = beta
        self._cache = 0

    @property
    def _mlearning_rate(self):
        return self._learning_rate

    @_mlearning_rate.setter
    def _mlearning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def _mdecay_rate(self):
        return self._decay_rate

    def _optimize(self, parameter, parameter_gradient):
        # For numerical stability
        eps = 10e-8
        # Calculate cache
        self._cache = self._beta*self._cache + (1-self._beta)*(parameter_gradient**2)
        # Update parameter and return
        return parameter + (-self._learning_rate*parameter_gradient)/((self._cache**0.5)+eps)
        
        
class Adam(Optimizer):
    '''
    This class implements adam optimization.
    '''
    def __init__(self, learning_rate, decay_rate=1, beta1=0.9, beta2=0.9):
        '''
        Parameters
        ----------
            learning_rate : float
            decay_rate : float
                Decay rate for leraning rate
            beta1 : float
                Should be between 0 to 1.
            beta2 : float
                Should be between 0 to 1.
        '''
        self._learning_rate = learning_rate
        self._decay_rate = decay_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._m = 0
        self._v = 0

    @property
    def _mlearning_rate(self):
        return self._learning_rate

    @_mlearning_rate.setter
    def _mlearning_rate(self, learning_rate):
        self._learning_rate = learning_rate

    @property
    def _mdecay_rate(self):
        return self._decay_rate

    def _optimize(self, parameter, parameter_gradient):
        # For numerical stability
        eps = 10e-8
        # Momentum
        self._m = self._beta1*self._m + (1-self._beta1)*parameter_gradient
        # RMS
        self._v = self._beta2*self._v + (1-self._beta2)*(parameter_gradient**2)
        # Update parameter
        return parameter + (-self._learning_rate*self._m)/((self._v**0.5)+eps)
