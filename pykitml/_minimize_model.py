from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt

import tqdm

class MinimizeModel(ABC):
    '''
    Abstract base class for all models that use gradients and minimize a cost function.
    '''

    def train(self, training_data, targets, batch_size, epochs, optimizer,
            testing_data=None, testing_targets=None, testing_freq=1, decay_freq=1):
        '''
        Trains the model on the training data, after training is complete, you can call
        :py:func:`plot_performance` to plot performance graphs.

        Parameters
        ----------
        training_data : numpy.array
            numpy array containing training data.
        targets : numpy.array
            numpy array containing training targets, corresponding to the training data.
        batch_size : int
            Number of training examples to use in one epoch, or
            number of training examples to use to estimate the gradient.
        epochs : int
            Number of epochs the model should be trained for.
        optimizer : any Optimizer object
            See :ref:`optimizers`
        testing_data : numpy.array
            numpy array containing testing data.
        testing_targets : numpy.array
            numpy array containing testing targets, corresponding to the testing data.
        testing_freq : int
            How frequently the model should be tested, i.e the model will be tested
            after every :code:`testing_freq` epochs. You may want to increase this to reduce 
            training time.
        decay_freq : int
            How frequently the model should decay the learning rate. The learning rate
            will decay after every :code:`decay_freq` epochs.

        Raises
        ------
        ValueError
            If :code:`training_data`, :code:`targets`, :code:`testing_data` or 
            :code:`testing_tagets` has invalid dimensions/shape. 
        '''
        # Dictionary for holding performance log
        self._performance_log = {}
        self._performance_log['epoch'] = []
        self._performance_log['cost_train'] = []
        if(testing_data is not None):
            self._performance_log['cost_test'] = []   

        # For hold total sum of gradients
        total_gradient = 0

        # Loop through each epoch
        with tqdm.trange(0, epochs, ncols=80, unit='epochs') as pbar:
            for epoch in pbar:
                # Loop through rest of the batch
                for batch in range(0, batch_size):
                    # Add the calculated gradients to the total
                    index = ((epoch*batch_size) + batch) % training_data.shape[0]
                    # Feedforward the input data
                    self.feedforward(training_data[index])
                    # Get gradient
                    total_gradient += self._backpropagate(targets[index])

                # After completing a batch, average the total sum of gradients and tweak the parameters
                self._mparams = optimizer._optimize(self._mparams, total_gradient/batch_size)

                # Zero the total 
                total_gradient = 0

                # Decay the learning rate
                if(epoch % decay_freq == 0):
                    optimizer._decay()

                # Log and print performance
                if((epoch+1)%testing_freq == 0):
                    # log epoch
                    self._performance_log['epoch'].append(epoch+1)
                    # get cost of the model on training data
                    cost_train = self.cost(training_data, targets)
                    pbar.set_postfix(cost=cost_train)
                    self._performance_log['cost_train'].append(cost_train)
                    # get cost of the model on testing data if it is provided
                    if(testing_data is not None):
                        cost_test = self.cost(testing_data, testing_targets)
                        self._performance_log['cost_test'].append(cost_test)

    def plot_performance(self):
        '''
        Plots logged performance data after training. Should be called after
        :py:func:`train` .

        Raises
        ------
        AttributeError
            If the model has not been trained, i.e :py:func`train` has
            not been called before.
        IndexError
            If :py:func:`train` failed.
        '''
        graph = self._performance_log

        # Window title
        plt.figure('Performance graph', figsize = (10, 7))

        # Plot average cost vs epochs on training data
        plt.plot(graph['epoch'], graph['cost_train'], label='Training data')
        
        # Plot average cost on testing data
        if('cost_test' in graph.keys()):
            plt.plot(graph['epoch'], graph['cost_test'], label='Test data')

        # Axis labels
        plt.ylabel('Average cost')
        plt.xlabel('No. of epochs')
        plt.legend()

        # Show the plot
        plt.show()

    def cost(self, testing_data, testing_targets):
        '''
        Tests the average cost of the model on the testing data passed to the
        function.

        Parameters
        ----------
        testing_data : numpy.array
            numpy array containing testing data.
        testing_targets : numpy.array
            numpy array containing testing targets, corresponding to the testing data.
        
        Returns
        -------
        cost : float
            The average cost of the model over the testing data.

        Raises
        ------
        ValueError
            If :code:`testing_data` or :code:`testing_tagets` has invalid dimensions/shape.      
        '''
        # Evalute over all the testing data and get outputs
        self.feedforward(testing_data)
        output_targets = self.get_output()

        # Calculate cost
        cost = np.sum(self._cost_function(output_targets, testing_targets))
        # Add regulerization
        cost += self._get_norm_weights()
        
        # Average the cost
        cost = cost/testing_data.shape[0]

        # return cost
        return round(cost, 2)

    @property
    @abstractmethod
    def _mparams(self):
        '''
        @property method to get model parameters
        '''
        pass
    
    @_mparams.setter
    @abstractmethod
    def _mparams(self, mparams):
        '''
        @params.setter method to set model parameters
        '''
        pass

    @abstractmethod
    def feedforward(self, input_data):
        '''
        Accepts input array and feeds it forward through the model.

        Parameters
        ----------
        input_data : numpy.array
            The input to feed forward through the model.

        Raises
        ------
        ValueError
            If the input data has invalid dimensions/shape.      

        Note
        ----
        This function only feed-forwards the input data, to get the output after calling this
        function use :py:func:`get_output` or :py:func:`get_output_one_hot` or :py:func:`result`
        '''
        pass

    @abstractmethod
    def get_output(self):
        '''
        Returns the output activations of the model.

        Returns
        -------
        numpy.array
            The output activations.
        '''
        pass

    @abstractmethod
    def _backpropagate(self, targets):
        '''
        This function calculates gradient of the cost function w.r.t all weights and 
        biases of the model by backpropagating the error through the model.

        Parameters
        ----------
        target : numpy.array
            The correct activations that the output layer should have.

        Returns
        -------
        gradient: numpy.array
            Numpy object array containing gradients, has same shape as self._mparams

        Raises
        ------
        ValueError
            If the input data has invalid dimensions/shape.

        Note
        ----
        You have to call :py:func:`~feedforward` before you call this function.
        ''' 
        pass

    @property
    @abstractmethod
    def _cost_function(self):
        '''
        Return the cost function used in the model.
        '''
        pass

    @abstractmethod
    def _get_norm_weights(self):
        '''
        Return the norm of all the regulerized parameters of the models
        multiplied by the regulerization parameter.
        '''
        pass


class Classifier(ABC):
    '''
    Mix-in class for classifier models.
    '''

    @abstractmethod
    def get_output(self):
        pass

    @abstractmethod
    def feedforward(self, input_data):
        pass

    def get_output_one_hot(self):
        '''
        Returns the output layer activations of the model as a one-hot array. A one-hot array
        is an array of bits in which only `one` of the bits is high/true. In this case, the
        corresponding bit to the neuron/node having the highest activation will be high/true.
        
        Returns
        -------
        numpy.array
            The one-hot output activations array. 
        '''
        # return output activations as onehot array
        activations = self.get_output()
        output = np.zeros(activations.shape[0])
        index = np.argmax(activations)
        output[index] = 1
        return output 

    def accuracy(self, testing_data, testing_targets):
        '''
        Tests the accuracy of the model on the testing data passed to the
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
           The accuracy of the model over the testing data i.e how many testing examples
           did the model predict correctly.
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

    def result(self):
        '''
        Returns index and activation of the node/neuron having the highest activation.

        Returns
        -------
        index : int
            The index(starts at zero) of the node/neuron having the highest activation.
        activation : float
            The activation of the node/neuron.
        '''
        # return the output layer activations along with the node/neuron with the most activation
        activations = self.get_output()
        index = np.argmax(activations)
        return index, activations[index] 