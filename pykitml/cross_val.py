import numpy as np

def cross_validate(inputs, outputs, folds=5):
    '''
    Python generator function for making K-fold cross validation easier.

    Parameters
    ----------
    inputs : numpy.array
        Inputs/features of training data.
    outputs : numpy.array
        Outputs/targets of training data.

    Yields
    ------
    train_inputs : numpy.array
        Training data containing inputs.
    train_outputs : numpy.array
        Training data containing outputs.
    test_inputs : numpy.array
        Testing data containing inputs.
    test_outputs : numpy.array
        Testing data containing outputs.

    Example
    -------
    >>> import numpy as np
    >>> import pykitml as pk
    >>> 
    >>> # Mock training data
    ... x = np.arange(30).reshape((10, 3))
    >>> y = x + 10
    >>> 
    >>> # 5-fold cross validation
    ... # Training data is split into 5 blocks, each block takes its turn
    ... # to be the test data.
    ... for train_x, train_y, test_x, test_y in pk.cross_validate(x, y, 5):
    ...     print(train_x)
    ...     print(train_y)
    ...     print(test_x)
    ...     print(test_y)
    '''
    size = inputs.shape[0]
    block_size = size//folds
    remainder = size%folds

    # Calculate block sizes
    def get_block_size(block):
        if(block < remainder): return block_size+1
        else: return block_size

    block_sizes = [get_block_size(block) for block in range(folds)]

    # Calculate block indices
    block_indices = [sum(block_sizes[:block]) for block in range(folds)]

    # Generate blocks
    def make_block(i, array):
        start = block_indices[i]
        end = block_indices[i]+block_sizes[i]
        return array[start:end]

    for i in range(folds):
        # Create testing data
        test_inputs, test_outputs = make_block(i, inputs), make_block(i, outputs)

        # Create training data
        train_blocks_inputs = [make_block(j, inputs) for j in range(folds) if(j!=i)]
        train_inputs = np.concatenate(train_blocks_inputs, axis=0)
        train_blocks_outputs = [make_block(j, outputs) for j in range(folds) if(j!=i)]
        train_outputs = np.concatenate(train_blocks_outputs, axis=0)

        yield train_inputs, train_outputs, test_inputs, test_outputs
