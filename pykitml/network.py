import numpy as np

from . import _minimize_model
from . import _functions

class NeuralNetwork(_minimize_model.MinimizeModel):
    '''
    This class implements the classic Feedforward Neural Network.
    '''

    def __init__(self, layer_sizes, config = 'leakyrelu-softmax-cross_entropy'):
        '''
        Parameters
        ----------
        layer_sizes : list
            A list of integers describing the number of layers and the number of neurons in each
            layer. For e.g. :code:`[784, 100, 100, 10]` describes a neural network with one input
            layer having 784 neurons, two hidden layers having 100 neurons each and a output layer
            with 10 neurons.
        config : str
            The config string describes what activation functions and cost function to use for
            the network. The string should contain three function names seperated with '-' 
            character and should follow the order:
            :code:`'<hidden_layer_activation_func>-<output_layer_activation_func>-<cost_function>'`.
            For e.g. :code:`'relu-softmax-cross_entropy'` tells the class to use `relu` as the
            activation function for input and hidden layers, `softmax` for output layer and
            `cross entropy` for the cost function.

            List of available activation functions:
            :code:`leakyrelu`, :code:`relu`, :code:`softmax`, :code:`tanh`, :code:`sigmoid`, :code:`identity`.

            List of available cost functions:
            :code:`mse` (Mean Squared Error), :code:`cross_entropy` (Cross Entropy).

        Raises
        ------
        AttributeError
            If invalid config string.
        '''
        # Get function names from the config string, the config string contains
        # list of functions to be used in the neural network, seperated with '-' character.
        # syntax: 'hidden_layers-output_layer-cost function', eg: 'relu-sigmoid-cross_entropy', 
        func_names = config.split('-')
        func_names_prime = [func_name + '_prime' for func_name in func_names]
        
        # Initilize functions
        self._activ_func = getattr(_functions, func_names[0])
        self._activ_func_prime = getattr(_functions, func_names_prime[0])
        self._output_activ_func = getattr(_functions, func_names[1])
        self._output_activ_func_prime = getattr(_functions, func_names_prime[1])
        self._cost_func = getattr(_functions, func_names[2])
        self._cost_func_prime = getattr(_functions, func_names_prime[2])

        # Store activation function names
        self._functs = func_names

        # Store layer sizes
        self._lsize = layer_sizes

        # Number of layers
        self._nlayers = len(layer_sizes)

        # Intialize parameters
        weights = [np.array([])] * self.nlayers
        biases = [np.array([])] * self.nlayers
        # Loop through each layer and initalize random weights and biases
        for l in range(1, self.nlayers):
            layer_size = layer_sizes[l]
            input_layer_size = layer_sizes[l-1]
            epsilon = np.sqrt(6)/(np.sqrt(layer_size) + np.sqrt(input_layer_size))
            weights[l] = np.random.rand(layer_size, input_layer_size)*2*epsilon - epsilon
            biases[l] = np.random.rand(layer_size) * 2 * epsilon - epsilon
        # Put parameters in numpy dtype=object array
        self._params = np.array(
            [np.array(weights, dtype=object), np.array(biases, dtype=object)],
            dtype=object
        )  

        # List of numpy arrays for storing temporary values
        self._weighted_sums = [np.array([])] * self.nlayers
        self._activations = [np.array([])] * self.nlayers

    def __repr__(self):
        return 'Layers: '+str(self._lsize)+' Config: '+str(self._functs)

    @property
    def _mparams(self):
        return self._params
    
    @_mparams.setter
    def _mparams(self, mparams):
        self._params = mparams

    @property
    def nlayers(self):
        '''
        The number of layers in the network.
        '''      
        return self._nlayers

    def feedforward(self, input_data):
        '''
        Accepts input array and feeds it forward through the network.

        Parameters
        ----------
        input_data : numpy.array
            The input to feed forward through the network.

        Raises
        ------
        ValueError
            If the input data has invalid dimensions/shape.      

        Note
        ----
        This function only feed-forwards the input data, to get the output after calling this
        function use :py:func:`get_output` or :py:func:`get_output_one_hot` or :py:func:`result` 
        '''
        # Constants
        W = 0 # Weights
        B = 1 # Biases

        # Set inputs
        self._activations[0] = input_data

        # Feed through hidden layers
        for l in range(1, self.nlayers-1):
            self._weighted_sums[l] = (self._activations[l-1]@self._params[W][l].transpose()) + self._params[B][l]
            self._activations[l] = self._activ_func(self._weighted_sums[l])

        # Feed thorugh output layer
        self._weighted_sums[-1] = (self._activations[-2]@self._params[W][-1].transpose()) + self._params[B][-1]
        self._activations[-1] = self._output_activ_func(self._weighted_sums[-1])

    def get_output(self):
        '''
        Returns the output layer activations of the network.

        Returns
        -------
        numpy.array
            The output layer activations.
        '''
        return self._activations[-1]

    def get_output_one_hot(self):
        '''
        Returns the output layer activations of the network as a one-hot array. A one-hot array
        is an array of bits in which only `one` of the bits is high/true. In this case, the
        corresponding bit to the neuron having the highest activation will be high/true.
        
        Returns
        -------
        numpy.array
            The one-hot output activations array. 
        '''
        # return output activations as onehot array
        output = np.zeros(self._lsize[-1])
        neuron_index = np.argmax(self._activations[-1])
        output[neuron_index] = 1
        return output 

    def result(self):
        '''
        Returns index and activation of the neuron having the highest activation.

        Returns
        -------
        neuron_index : int
            The index(starts at zero) of the neuron having the highest activation.
        activation : float
            The activation of the neuron.
        '''
        # return the output layer activations along with the neuron with the most activation
        neuron_index = np.argmax(self._activations[-1])
        return neuron_index, self._activations[-1][neuron_index] 

    def _backpropagate(self, target):
        '''
        This function calculates gradient of the cost function w.r.t all weights and 
        biases of the network by backpropagating the error through the network.

        Parameters
        ----------
        target : numpy.array
            The correct activations that the output layer should have.

        Returns
        -------
        dc_dw : list
            List containing numpy arrays, each numpy array corresponding to the gradient of
            weights of a layer.
        dc_db : list
            List containing numpy arrays, each numpy array corresponding to the gradient of
            biases of a layer.

        Raises
        ------
        ValueError
            If the input data has invalid dimensions/shape.

        Note
        ----
        You have to call :py:func:`~feedforward` before you call this function.
        '''
        # Constants
        W = 0 # Weights        
        
        # Used to hold activation_function'(z[l]) where z[l] = w[l]*a[l-1] + b[l] 
        da_dz = [np.array([])] * self.nlayers

        # Used to hold partial derivatives of cost function w.r.t parameters
        dc_dw = [np.array([])] * self.nlayers
        dc_db = [np.array([])] * self.nlayers
        
        # Calculate activation_function'(z)
        def calc_da_dz(l):
            da_dz[l] = self._activ_func_prime(self._weighted_sums[l], self._activations[l])
        
        # Calculate the partial derivatives of the cost w.r.t all the biases of layer 
        # 'l' (NOT for output layer)
        def calc_dc_db(l):
            dc_db[l] = (dc_db[l+1] @ self._params[W][l+1]) * da_dz[l]

        # Calculate the partial derivatives of the cost w.r.t all the weights of layer 'l'
        def calc_dc_dw(l):
            dc_dw[l] = np.multiply.outer(dc_db[l], self._activations[l-1])

        # Calculate the partial derivatives of the cost function w.r.t the ouput layer's 
        # activations, weights, biases
        da_dz[-1] = self._output_activ_func_prime(self._weighted_sums[-1], self._activations[-1])
        dc_db[-1] = self._cost_func_prime(self._activations[-1], target) * da_dz[-1]
        calc_dc_dw(-1)

        # Calculate the partial derivatives of the cost function w.r.t the hidden layers'
        # activations, weights, biases
        for l in range(self.nlayers - 2, 0, -1):
            calc_da_dz(l)
            calc_dc_db(l)
            calc_dc_dw(l)

        # return gradient
        return np.array(
            [np.array(dc_dw, dtype=object), np.array(dc_db, dtype=object)],
            dtype=object
        )

    def _get_grad(self, input_data, target):
        # Feedforward the input data
        self.feedforward(input_data)
        # Get gradients and return
        return self._backpropagate(target)

    def cost(self, testing_data, testing_targets):
        # Evalute over all the testing data and get outputs
        self.feedforward(testing_data)
        output_targets = self.get_output()

        # Calculate average cost
        cost = np.sum(self._cost_func(output_targets, testing_targets))/testing_data.shape[0]

        # return cost
        return round(cost, 2)

    def accuracy(self, testing_data, testing_targets):
        '''
        Tests the accuracy of the network on the testing data passed to the
        function. This function should be only used for clssification.

        Parameters
        ----------
        testing_data : numpy.array
            numpy array containing testing data.
        testing_targets : numpy.array
            numpy array containing testing targets, corresponding to the testing data.
        
        Returns
        -------
        accuracy : float
           The accuracy of the network over the testing data i.e how many testing examples
           did the network predict correctly.
        '''
        # Evalute over all the testing data and get outputs
        self.feedforward(testing_data)
        output_targets = self.get_output()

        # Create a onehot array from outputs
        output_one_hot = np.zeros(output_targets.shape)
        output_one_hot[np.arange(output_targets.shape[0]), np.argmax(output_targets, axis=1)] = 1

        # Calculate how many examples it classified correctly
        no_correct = (testing_targets == output_one_hot).all(axis=1).sum()
        accuracy = (no_correct/testing_data.shape[0]) * 100

        # return accuracy
        return round(accuracy, 2)


