import os
import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import adult

import pytest

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

@pytest.mark.skip(reason='Will take too long')
def test_adult_tree():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import adult

    # Download the dataset 
    if(not os.path.exists('adult.data.pkl')): adult.get()

    # Load adult data set
    inputs_train, outputs_train, inputs_test, outputs_test = adult.load() 
    outputs_train = pk.onehot(outputs_train)
    outputs_test = pk.onehot(outputs_test)

    # Create model
    ftypes = [
        'continues', 'categorical', 'continues', 'categorical',
        'categorical', 'categorical', 'categorical', 'categorical', 'categorical',
        'continues', 'continues', 'continues', 'categorical'
    ]
    tree_adult_classifier = pk.DecisionTree(13, 2, max_depth=6, feature_type=ftypes)

    # Train
    tree_adult_classifier.train(inputs_train, outputs_train)

    # Save it
    pk.save(tree_adult_classifier, 'tree_adult_classifier.pkl')

    # Print accuracy
    accuracy = tree_adult_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = tree_adult_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    tree_adult_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['False', 'True'])

    # Plot descision tree
    tree_adult_classifier.show_tree()

    # Assert accuracy
    assert (tree_adult_classifier.accuracy(inputs_test, outputs_test)) >= 84

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_adult_tree)
        profiler.dump_stats('test_adult_tree.dat') 
    except AssertionError:
        pass