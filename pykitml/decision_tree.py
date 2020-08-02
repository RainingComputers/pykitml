from itertools import combinations

import numpy as np
from graphviz import Digraph 
import tqdm

from ._regressor import Regressor
from ._classifier import Classifier
from ._exceptions import _valid_list, InvalidFeatureType

from . import _functions


def condition(column, split, ftype):
    '''
    Returns boolean array for splitting the dataset.

    Parameters
    ----------
    column : numpy.array
        The column of the dataset.
    split : int or tuple.
        The point or categories to split the dataset with.
    ftype : string
        The type of feature, :code:`'continues'`, :code:`'ranked'`,
        or :code:`'categorical'`. 
    '''
    if(ftype == 'ranked' or ftype == 'continues'):
        return column < split
    elif(ftype == 'categorical'):
        cond = np.full(column.shape, True, dtype=bool)
        for category in split:
            cond = np.logical_and(cond, (column==category))
        return cond


class _Node:
    def __init__(self, split, col, gini_index, nindex, feature_type):
        '''
        The node class.

        Parameters
        ----------
        split : int or tuple
            The point or categories to split the dataset with.
        col : int
            The features col to apply the split on.
        gini_index : float
            The gini index for this node.
        nindex : int
            ID for this node.
        feature_type : string
            The type of feature, :code:`'continues'`, :code:'`'ranked'`,
            or :code:`'categorical'` 
        '''
        # Condition that will split the data
        self._split = split
        self._col = col
        self._ftype = feature_type
        self._gini = round(gini_index, 4)
        self._index = nindex

        # Reference to two nodes
        self.right_node = None
        self.left_node = None

    @property
    def leaf(self):
        return False

    def decision(self, input_data):
        '''
        Splits the dataset and passes it to subnodes. The inputs
        travel till the reach a leaf node and backtrack as outputs.
        '''
        # Make sure input data is 2d
        if(input_data.ndim == 1): input_data = np.array([input_data])

        # Condition
        cond = condition(input_data[:, self._col], self._split, self._ftype)
        
        # Split data, travel to sub nodes and get output
        left_output = self.left_node.decision(input_data[cond]) 
        right_output = self.right_node.decision(input_data[~cond])
        
        # Return output
        outputs = np.zeros((input_data.shape[0], left_output.shape[1]))
        outputs[cond] = left_output
        outputs[~cond] = right_output
        return outputs

    def __str__(self):
        if(self._ftype == 'ranked' or self._ftype == 'continues'):
            if_str = 'Feature-'+str(self._col)+' < '+str(self._split)
        elif(self._ftype == 'categorical'):
            cat_str = str(self._split).replace(',', ' or')
            if_str = 'Feature-'+str(self._col)+' = '+cat_str
        
        return if_str+'\nNode-'+str(self._index)


class _Leaf:
    def __init__(self, term_val, gini_index, nindex):
        '''
        Leaf node or terminal node.

        Parameters
        ----------
        term_val : numpy.array
            The output/probabilities for this node.
        gini_index : float
            Gini index for this node.
        nindex : int
            ID for this node.
        '''
        self._term_val = np.round(term_val, 2)
        self._samples = None
        self._gini = round(gini_index, 4)
        self._index = nindex

    @property
    def leaf(self):
        return True

    def decision(self, input_data):
        '''
        Return outputs/probabilities. Starts the backtracking
        process.
        '''
        self._samples = input_data.shape[0]
        return np.tile(self._term_val, (input_data.shape[0], 1))

    def __str__(self):
        return  str(self._term_val)+'\nNode - '+str(self._index)


class DecisionTree(Classifier, Regressor):
    '''
    Implements Decision Tree model.
    '''

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
            The maximum depth the tree can grow to. Prevents from 
            overfitting (somewhat).
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
        self._node_count = 0
        self._min_split = min_split
        self._regression = regression
        self._max_splits_eval = max_splits_eval

        # Check if given feature types are valid
        valid_ftypes = ['continues', 'ranked', 'categorical']
        if not _valid_list(feature_type, valid_ftypes):
            raise InvalidFeatureType

        # The columns on which the tree will train on
        # This variable can be overridden in a child class to ignore
        # certain columns of the input data while training
        self._cols_train = list(range(input_size))

        # Can be overridden in child class to suppress progressbar while training
        self._pbardis = False

        # Tree nodes
        self._root_node = None

        # Outputs
        self._output = None

    @property
    def _out_size(self):
        return self._output_size

    def train(self, inputs, outputs):
        '''
        Trains the model on the training data.

        Parameters
        ----------
        training_data : numpy.array
            numpy array containing training data.
        targets : numpy.array
            numpy array containing training targets, corresponding to the training data.

        Raises
        ------
        numpy.AxisError
            If output_size is less than two. Use :py:func:`pykitml.onehot` to change
            0/False to [1, 0] and 1/True to [0, 1] for binary classification.
        '''
        print('Training Model...')
        
        # Convert outputs from onehot to values
        if(not self._regression):
            outputs_train = np.argmax(outputs, axis=1)
        else:
            outputs_train = outputs
        
        # Grow the tree
        pbar = tqdm.tqdm(total=inputs.shape[0], ncols=80, unit='expls', disable=self._pbardis)
        self._iterative_grow(inputs, outputs_train, pbar)
        # Close progress bar
        pbar.close()

    def feed(self, input_data):
        self._output = self._root_node.decision(input_data)

    def get_output(self):
        return self._output.squeeze()

    def _get_splits(self, column, ftype):
        '''
        Given input column of a dataset and the type of feature,
        generates all possible points/categories to split the dataset on.

        Parameters
        ----------
        column : int
            The column of the dataset.
        ftype : str
            The type of feature. Can be :code:`continues`, :code:`continues`, 
            :code:'`ranked`, or :code:`categorical`.,
        '''
        if(ftype == 'ranked' or ftype == 'continues'):
            # All possible values to split the dataset
            splits = np.unique(column) 
            # If there are too many candidates, randomly pick some candidates
            if(splits.shape[0] > self._max_splits_eval):
                splits = np.random.choice(splits, size=self._max_splits_eval, replace=False)

            return splits
        elif(ftype == 'categorical'):
            # All the possible 'or' combinations as a list of tuples
            categories = np.unique(column).tolist()
            combs = list(combinations(categories, len(categories)-1))
            categories = [tuple([x]) for x in categories]
            return combs+categories

    def _iterative_grow(self, inputs, outputs, pbar):
        '''
        This method iteratively generates nodes of the tree.
        '''
        self._node_count = 0
        self._root_node = None

        nodes_to_build = []

        if(not self._regression):
            value, score = self._gini_index(outputs)
        else:
            value, score = self._regression_score(outputs)

        nodes_to_build.append(
            {
                'node':None,
                'col':-1,
                'inputs':inputs,
                'outputs':outputs,
                'value':value,
                'score':score,
                'depth':1,
                'left':None
            }
        )

        # Start building the tree
        while(len(nodes_to_build) != 0):
            # Pop build parameters of stack
            node_build = nodes_to_build.pop()
            parent_node = node_build['node']
            col = node_build['col']
            inputs = node_build['inputs']
            outputs = node_build['outputs']
            value = node_build['value']
            score = node_build['score']
            depth = node_build['depth']
            left = node_build['left']

            def assign_node(node):
                if(self._node_count == 1): self._root_node = node
                elif(left): parent_node.left_node = node
                elif(not left): parent_node.right_node = node

            # Stopping condition
            if(depth == self._max_depth or inputs.shape[0] < self._min_split): 
                self._node_count+=1
                pbar.update(inputs.shape[0])
                node = _Leaf(value, score, self._node_count)
                assign_node(node)
                continue

            # Get the columns to iterate
            cols_train = [i for i in self._cols_train if i != col]

            # Keep track of least score
            min_score = float('inf')
            min_split = None
            min_col = None
            min_inputs_left = None
            min_inputs_right = None
            min_outputs_left = None
            min_outputs_right = None
            min_vel_left = None
            min_val_right = None
            min_score_left = None
            min_score_right = None

            # Generate data splits, get best split condition
            for col in cols_train:
                # Loop through each possible split
                splits = self._get_splits(inputs[:, col], self._ftype[col])
                for split in splits:
                    # Generate split condition
                    cond = condition(inputs[:, col], split, self._ftype[col])

                    # Split the data based on condition
                    inputs_left, outputs_left = inputs[cond, :], outputs[cond]
                    inputs_right, outputs_right = inputs[~cond, :], outputs[~cond]

                    # Continue if no split
                    if(inputs_left.shape[0]==0 or inputs_right.shape[0]==0):
                        continue

                    # Calculate score for the split
                    if(not self._regression):
                        # Get gini index for this column and split
                        weight_left = inputs_left.shape[0]/inputs.shape[0]
                        weight_right = inputs_right.shape[0]/inputs.shape[0]
                        val_left, score_left = self._gini_index(outputs_left)
                        val_right, score_right = self._gini_index(outputs_right)
                        score_col = weight_left*score_left + weight_right*score_right
                    else:
                        # For regression, calculate mse
                        val_left, score_left = self._regression_score(outputs_left)
                        val_right, score_right = self._regression_score(outputs_right)
                        score_col = score_left + score_right


                    # Track minimun value
                    if(score_col < min_score):
                        min_score = score_col
                        min_split = split
                        min_col = col
                        min_inputs_left = inputs_left
                        min_inputs_right = inputs_right
                        min_outputs_left = outputs_left
                        min_outputs_right = outputs_right
                        min_vel_left = val_left
                        min_val_right = val_right
                        min_score_left = score_left
                        min_score_right = score_right

            # Create node
            self._node_count+=1
            pbar.set_postfix(nodes=self._node_count)

            # If best split is not better than current node's gini index, stop
            # Create terminal node or leaf
            # If maxdepth has been exceeded, create leaf and return
            if(score <= min_score): 
                pbar.update(inputs.shape[0])
                node = _Leaf(value, score, self._node_count)
            else:
                # Create node, split data further
                node = _Node(min_split, min_col, min_score, self._node_count, self._ftype[min_col])

            # Assign the node to parent node
            assign_node(node)

            # If leaf node, no more nodes to build
            if(node.leaf): continue

            # Create left sub node
            nodes_to_build.append(
                {
                    'node':node,
                    'col':min_col,
                    'inputs':min_inputs_left,
                    'outputs':min_outputs_left,
                    'value':min_vel_left,
                    'score':min_score_left,
                    'depth':(depth+1),
                    'left':True
                }
            )

            # Create right sub node
            nodes_to_build.append(
                {
                    'node':node,
                    'col':min_col,
                    'inputs':min_inputs_right,
                    'outputs':min_outputs_right,
                    'value':min_val_right,
                    'score':min_score_right,
                    'depth':(depth+1),
                    'left':False
                }
            )


    def _gini_index(self, outputs):
        '''
        Given the outputs of a split dataset, calculates gini index,
        i.e. how 'pure' the dataset is. 0 for most pure and 1 for least
        pure.
        '''
        p_ci = np.zeros((self._out_size))
        # Get their probabilities P(Ci)
        bincnt = np.bincount(outputs)/outputs.shape[0]
        p_ci[0:bincnt.shape[0]] = bincnt
        # gini index gi = 1-sum(P(Ci)**2)
        return p_ci, 1-(p_ci**2).sum()

    def _regression_score(self, outputs):
        '''
        Given the outputs of a split dataset, calculates the mse error
        for the decision stump's average output value.
        '''
        avg_val = np.mean(outputs, axis=0)
        error = ((outputs-avg_val)**2).sum()
        return avg_val, error

    def show_tree(self):
        '''
        Draws a visualization/graph of the tree.
        '''
        print('Drawing Tree...')

        def walk(pbar, g, node):
            # Exit condition
            if(node.leaf): return
            # Draw left node
            g.edge(str(node), str(node.left_node), label='True')
            # Draw right node
            g.edge(str(node), str(node.right_node), label='False')
            # Update progress bar
            pbar.update(2)
            # Draw subnodes
            walk(pbar, g, node.left_node)
            walk(pbar, g, node.right_node)

        pbar = tqdm.tqdm(total=self._node_count, ncols=80, unit='nodes')
        pbar.update()
        g = Digraph('Tree', format='png')
        walk(pbar, g, self._root_node)

        g.view()
