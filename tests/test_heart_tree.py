from pykitml.testing import pktest_graph, pktest_nograph

@pktest_graph
def test_heart_tree():
    import os.path
    
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import heartdisease

    # Download the dataset 
    if(not os.path.exists('heartdisease.pkl')): heartdisease.get()

    # Load heart data set
    inputs, outputs = heartdisease.load()
    outputs = pk.onehot(outputs)

    # Create model
    ftypes = [
        'continues', 'categorical', 'categorical',
        'continues', 'continues', 'categorical', 'categorical',
        'continues', 'categorical', 'continues', 'categorical',
        'categorical', 'categorical'
    ]
    tree_heart_classifier = pk.DecisionTree(13, 2, max_depth=6, feature_type=ftypes)

    # Train
    tree_heart_classifier.train(inputs, outputs)

    # Save it
    pk.save(tree_heart_classifier, 'tree_heart_classifier.pkl')

    # Print accuracy
    accuracy = tree_heart_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    tree_heart_classifier.confusion_matrix(inputs, outputs, 
        gnames=['False', 'True'])

    # Plot descision tree
    tree_heart_classifier.show_tree()

    # Assert accuracy
    assert (tree_heart_classifier.accuracy(inputs, outputs)) >= 94


if __name__ == '__main__':
    try:
        test_heart_tree.__wrapped__()
    except AssertionError:
        pass