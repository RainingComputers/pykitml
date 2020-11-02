from abc import ABC, abstractmethod
import itertools

import numpy as np
import matplotlib.pyplot as plt

import tqdm

from . import _heatmap
from . import preprocessing

class Classifier(ABC):
    '''
    Mix-in class for classifier models.
    '''

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

    @property
    @abstractmethod
    def _out_size(self):
        '''
        Returns number of nodes/neurons in the output layer.
        '''
        pass

    def get_output_onehot(self):
        '''
        Returns the output layer activations of the model as a one-hot array. A one-hot array
        is an array of bits in which only `one` of the bits is high/true. In this case, the
        corresponding bit to the neuron/node having the highest activation will be high/true.
        
        Returns
        -------
        numpy.array
            The one-hot output activations array. 
        '''
        output_targets = self.get_output()

        if(self._out_size == 1):
            # For binary classification
            return np.where(output_targets>0.5, 1, 0)
        elif(output_targets.ndim == 1):
            # If output is a vector/1D array, axis=1 will not work
            index = np.argmax(output_targets)
            output_onehot = np.zeros((self._out_size))
            output_onehot[index] = 1
        else:
            # Create a onehot array from outputs
            output_onehot = np.zeros(output_targets.shape)
            output_onehot[np.arange(output_targets.shape[0]), np.argmax(output_targets, axis=1)] = 1

        # Return one hot array
        return output_onehot

    def accuracy(self, testing_data, testing_targets):
        '''
        Tests the accuracy of the model on the testing data passed to the
        function. This function should be only used for classification.

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
        self._on_test_start()

        # Evalute over all the testing data and get outputs
        self.feed(testing_data)

        # Create a onehot array from outputs
        output_onehot = self.get_output_onehot()

        # Calculate how many examples it classified correctly
        if(self._out_size == 1):
            no_correct = (testing_targets == output_onehot).sum()
        else:    
            no_correct = (testing_targets == output_onehot).all(axis=1).sum()
        
        # Calculate accuracy
        accuracy = (no_correct/testing_data.shape[0]) * 100

        self._on_test_end()
        # return accuracy
        return round(accuracy, 2) 

    def confusion_matrix(self, test_data, test_targets, gnames=[], plot=True,):   
        '''
        Returns and plots confusion matrix on the given test data.

        Parameters
        ----------
        test_data : numpy.array
            Numpy array containing test data
        test_targets : numpy.array
            Numpy array containing the targets corresponding to the test data.
        plot : bool
            If set to false, will not plot the matrix. Default is true.
        gnames : list
            List of string names for each class/group.

        Returns
        -------
        confusion_matrix : numpy.array
            The confusion matrix. 
        '''
        print('Creating Confusion Matrix...')
        self._on_test_start()

        # feed the data
        self.feed(test_data)

        # Get output
        outputs = self.get_output_onehot()

        # Binary classification
        if(self._out_size == 1):
            # Number of groups
            ngroups = 2
            # Column/Row labels
            labels = ['False', 'True']
            # Split outputs to two groups
            outputs = preprocessing.onehot(outputs)
            targets = preprocessing.onehot(test_targets)
            # Prevent bugs that show up when outputs are all zeros
            if(outputs.shape[1] == 1):
                outputs = np.pad(outputs, ((0, 0), (0 ,1)), 'constant', constant_values=0)

        # Multiclass classification
        else:
            # Number of groups
            ngroups = self._out_size
            # Column/Row labels
            if(len(gnames) != 0):
                labels = gnames
            else:
                labels = [str(x) for x in range(0, ngroups)]
            # Targets
            targets = test_targets
        
        # Confusion matrix
        conf_mat = np.zeros((ngroups, ngroups))

        # Loop through every possibility and count them
        pairs = list(itertools.product(range(0, ngroups), repeat=2))
        for predicted, actual in tqdm.tqdm(pairs, ncols=80, unit='pairs'):
            # Make one hot vector for predicted
            pred_vec = np.zeros((ngroups))
            pred_vec[predicted] = 1

            # Make one hot vector for actual
            act_vec = np.zeros((ngroups))
            act_vec[actual] = 1

            # Count
            out_count = np.all(outputs == pred_vec, axis=1)
            target_count  = np.all(targets == act_vec, axis=1)
            tot_count = np.logical_and(out_count, target_count).sum()
            conf_mat[predicted][actual] = tot_count

        # Plot the confusion matrix
        if(plot):
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(10, 7))
            im, _ = _heatmap.heatmap(conf_mat, labels, labels, ax=ax,
                            cmap="YlGn", cbarlabel="Count")
            _heatmap.annotate_heatmap(im, valfmt="{x:.0f}")

            # Labels
            fig.canvas.set_window_title('Confusion Matrix') 
            ax.set_xlabel('Actual')
            ax.xaxis.set_label_position('top') 
            ax.set_ylabel('Predicted')

            # Show
            plt.show()

        # return
        self._on_test_end()
        return conf_mat

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