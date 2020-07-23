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
            :code:`testing_targets` has invalid dimensions/shape. 
        '''
        print('Training Model...')

        # Dictionary for holding performance log
        self._performance_log = {}
        self._performance_log['epoch'] = []
        self._performance_log['cost_train'] = []
        self._performance_log['learning_rate'] = []
        if(testing_data is not None):
            self._performance_log['cost_test'] = []   

        self._init_train(batch_size)

        # Loop through each epoch
        pbar = tqdm.trange(0, epochs, ncols=80, unit='epochs')
        for epoch in pbar:
            total_gradient = self._get_batch_grad(epoch, batch_size, training_data, targets)

            # After completing a batch, average the total sum of gradients and tweak the parameters
            self._mparams = optimizer._optimize(self._mparams, total_gradient)

            # Decay the learning rate
            if((epoch+1) % decay_freq == 0): optimizer._decay()

            # Log and print performance
            if((epoch+1)%testing_freq == 0):
                # log epoch
                self._performance_log['epoch'].append(epoch+1)
                
                # log learning rate
                learning_rate = optimizer._learning_rate
                self._performance_log['learning_rate'].append(learning_rate)
                
                # get cost of the model on training data
                cost_train = self.cost(training_data, targets)
                pbar.set_postfix(cost=cost_train)
                self._performance_log['cost_train'].append(cost_train)
                
                # get cost of the model on testing data if it is provided
                if(testing_data is None): continue
                cost_test = self.cost(testing_data, testing_targets)
                self._performance_log['cost_test'].append(cost_test)

        # Close progress bar
        pbar.close()

    def _get_batch_grad(self, epoch, batch_size, training_data, targets):
        '''
        Calculates the sum of all the gradients fot the given batch

        Parameters
        ----------
        epoch : int
            Current epoch in the training process.
        batch_size : int
            Size of the batch.
        training_data : numpy.array
            numpy array containing training data.
        targets : numpy.array
            numpy array containing training targets, corresponding to the training data.
        '''
        # Total gradient for the batch 
        total_gradient = 0
        
        # feed the batch
        start_index = (epoch*batch_size)%training_data.shape[0]
        end_index = start_index+batch_size
        indices = np.arange(start_index, end_index) % training_data.shape[0]
        self.feed(training_data[indices])

        if(self.bptt):
            return self._backpropagate(None, targets[indices])
        else:
            # Loop through the batch
            for example in range(0, batch_size):
                # Add the calculated gradients to the total
                index = ((epoch*batch_size) + example)%training_data.shape[0]
                # Get gradient
                total_gradient += self._backpropagate(example, targets[index])

            # return the total
            return total_gradient/batch_size

    def plot_performance(self):
        '''
        Plots logged performance data after training. 
        Should be called after :py:func:`train`.

        Raises
        ------
        AttributeError
            If the model has not been trained, i.e :py:func:`train` has
            not been called before.
        IndexError
            If :py:func:`train` failed.
        '''
        graph = self._performance_log

        # Window title
        plt.figure('Performance graph', figsize=(10, 7))

        # First subplot
        plt.subplot(2, 1, 1)

        # Plot average cost vs epochs on training data
        plt.plot(graph['epoch'], graph['cost_train'], label='Training data')
        
        # Plot average cost on testing data
        if('cost_test' in graph.keys()):
            plt.plot(graph['epoch'], graph['cost_test'], label='Test data')

        # Axis labels
        plt.ylabel('Average cost')
        plt.xlabel('No. of epochs')
        plt.legend()

        # Second subplot
        plt.subplot(2, 1, 2)

        # Plot learning rate vs epochs
        plt.plot(graph['epoch'], graph['learning_rate'], label='Learning Rate')

        # Axis labels
        plt.ylabel('Learning Rate')
        plt.xlabel('No. of epochs')
        plt.legend()        

        # Show the plot
        plt.show()

    def _init_train(self, batch_size):
        '''
        This function will be called before training starts.
        Override this function to do initialization.
        
        Parameters
        ----------
        batch_size : int
            Size of the batch.       
        '''
        pass

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
        self._on_test_start()

        # Evalute over all the testing data and get outputs
        self.feed(testing_data)
        output_targets = self.get_output()

        # Calculate cost
        cost = np.sum(self._cost_function(output_targets, testing_targets))
        # Add regularization
        cost += self._get_norm_weights()
        
        # Average the cost
        cost = cost/testing_data.shape[0]

        self._on_test_end()

        # return cost
        return round(cost, 2)

    def _on_test_start(self):
        '''
        This method will be called before testing, 
        override this method needed.
        '''
        pass

    def _on_test_end(self):
        '''
        This method will be called after testing, 
        override this method needed.
        '''
        pass

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

    @property
    @abstractmethod
    def bptt(self):
        '''
        Return True if the model requires BPTT
        (Backpropagation through time), otherwise return false
        '''
        pass

    @abstractmethod
    def feed(self, input_data):
        '''
        Accepts input array and feeds it to the model.

        Parameters
        ----------
        input_data : numpy.array
            The input to feed the model.

        Raises
        ------
        ValueError
            If the input data has invalid dimensions/shape.      

        Note
        ----
        This function only feeds the input data, to get the output after calling this
        function use :py:func:`get_output` or :py:func:`get_output_onehot`
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
    def _backpropagate(self, index, targets):
        '''
        This function calculates gradient of the cost function w.r.t all weights and 
        biases of the model by backpropagating the error through the model.

        Parameters
        ----------
        index : int
            Index of the example in the batch that was fed using feed.
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
        You have to call :py:func:`~feed` before you call this function.
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
        Return the norm of all the regularized parameters of the models
        multiplied by the regularization parameter.
        '''
        pass

    