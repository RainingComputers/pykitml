import os
import random
import multiprocessing as mp
from math import ceil
from contextlib import redirect_stdout

import numpy as np
import tqdm

from . import _shared_array
from ._regressor import Regressor
from ._classifier import Classifier
from . import decision_tree

class _RandomTree(decision_tree.DecisionTree):
    def __init__(self, input_size, output_size, num_features, feature_type=[], 
        max_depth=6, min_split=2, max_splits_eval=100, regression=False):
        # Initialize parent class
        super(_RandomTree, self).__init__(input_size, output_size, feature_type, max_depth,
            min_split, max_splits_eval, regression)

        # Select only a few random columns of the dataset for training
        self._cols_train = np.random.choice(input_size, num_features, replace=False)

        # Disable progress bar
        self._pbardis = True
        

class RandomForest(Classifier, Regressor):
    def __init__(self, input_size, output_size, feature_type=[], max_depth=6, min_split=2,
        max_splits_eval=100, regression=False):
        '''
        Parameters
        ----------
        input_size : int
            Size of input data or number of input features.
        output_size : int
            Number of categories or groups.
        feature_type : list
            List of string describing the type of feature for
            each column. Can be :code:`'continues'`, 
            :code:`'ranked'`, or :code:`'categorical'`.
        max_depth : int
            The maximum depth the trees can grow to.
        min_split : int
            The minimum number of data points a node should have to get 
            split.
        max_splits_eval : int
            The maximum number of split points to evaluate for an 
            attribute. If the number of candidate split points exceed
            this, :code:`max_splits_eval` split candidates will be
            randomly sampled from the candidates and only the sampled
            ones will be evaluated from finding the best split point.
        regression : bool
            If the tree is being trained on a regression problem.

        Raises
        ------
        InvalidFeatureType
            Invalid/Unknown feature type. Can only be :code:`'continues'`, 
            :code:`'ranked'`, or :code:`'categorical'`.
        '''
        # Save values
        self._input_size = input_size
        self._output_size = output_size
        self._ftype = feature_type
        self._max_depth = max_depth
        self._min_split = min_split
        self._regression = regression
        self._max_splits_eval = max_splits_eval

        # List to store trees in
        self._trees = []

        # Outputs
        self._output = None

    @property
    def _out_size(self):
        return self._output_size
    
    @property
    def trees(self):
        '''
        A list of decision trees used in the forest.
        '''
        return self._trees

    def train(self, inputs, outputs, num_trees=100, num_feature_bag=None):
        '''
        Trains the model on the training data.

        Parameters
        ----------
        training_data : numpy.array
            numpy array containing training data.
        targets : numpy.array
            numpy array containing training targets, corresponding to the training data.
        num_trees : int
            Number of trees to grow.
        num_feature_bag : int or None
            Number of random features to select when growing
            a tree. If :code:`None` (default), :code:`ceil(sqrt(input_size))`
            is chosen for classification and :code:`int(input_size/3)` for regression.

        Raises
        ------
        numpy.AxisError
            If output_size is less than two. Use :py:func:`pykitml.onehot` to change
            0/False to [1, 0] and 1/True to [0, 1] for binary classification.
        '''
        print('Training Model...')

        # Number of features to bag/choose for each tree
        if(num_feature_bag is None): 
            if(not self._regression):
                num_feature_bag = ceil(np.sqrt(self._input_size))
            else:
                num_feature_bag = int(self._input_size/3)
        
        def train_trees(input_q, ret_q, inputs_sh, inputs_shape, outputs_sh, outputs_shape):
            # Retrive numpy arrays from multiprocessing arrays
            inputs = _shared_array.shm_as_ndarray(inputs_sh, inputs_shape)
            output = _shared_array.shm_as_ndarray(outputs_sh, outputs_shape)
            
            # Supress print statements
            with redirect_stdout(open(os.devnull, 'w')):
                while(True):
                    # Get tree from input queue
                    try:
                        tree = input_q.get(block=False)
                    except mp.queues.Empty:
                        break

                    # Create bootstraped datset
                    indices = np.random.choice(inputs.shape[0], inputs.shape[0])
                    bootstrapped_inputs = inputs[indices]
                    bootstrapped_outputs = outputs[indices]

                    # Grow the tree
                    tree.train(bootstrapped_inputs, bootstrapped_outputs)  

                    # Put the trained tree in output queue
                    ret_q.put(tree)  

        # Create queues
        input_q = mp.Queue()
        ret_q = mp.Queue()

        # Initialize input queue
        for i in range(num_trees):
            # Create tree
            tree = _RandomTree(self._input_size, self._output_size, num_feature_bag, 
                self._ftype, self._max_depth, self._min_split, self._max_splits_eval,
                self._regression)
            # Put it in queue
            input_q.put(tree)

        # Create shared multiprocess array for inputs and outputs
        inputs_sh = _shared_array.ndarray_to_shm(inputs)
        outputs_sh = _shared_array.ndarray_to_shm(outputs)

        # Start multiprocess
        for i in range(os.cpu_count()):
            p = mp.Process(
                target=train_trees, args=(input_q, ret_q, 
                    inputs_sh, inputs.shape, outputs_sh, outputs.shape)
            )
            p.start()

        # Progress bar and append trained trees to list
        pbar = tqdm.tqdm(total=num_trees, ncols=80, unit='trees')
        
        while(len(self._trees) != num_trees):
            tree = ret_q.get()
            self._trees.append(tree)
            pbar.update()
        
        # Return if done
        pbar.close()

    def feed(self, input_data):
        # Loop through all the trees and total their outputs
        total = 0
        for tree in self._trees:
            tree.feed(input_data)
            total += tree.get_output()
        
        # Average
        self._output = total/len(self._trees)
        
    
    def get_output(self):
        return self._output.squeeze()

        

