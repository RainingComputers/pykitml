from pykitml.testing import pktest_graph, pktest_nograph

@pktest_graph
def test_banknote_tree():
    import os

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import banknote

    # Download the dataset 
    if(not os.path.exists('banknote.pkl')): banknote.get()

    # Load heart data set
    inputs_train, outputs_train, inputs_test, outputs_test = banknote.load()
    
    # Change 0/False to [1, 0]
    # Change 1/True to [0, 1]
    outputs_train = pk.onehot(outputs_train)
    outputs_test = pk.onehot(outputs_test)

    # Create model
    ftypes = ['continues']*4
    tree_banknote_classifier = pk.DecisionTree(4, 2, max_depth=7, feature_type=ftypes)

    # Train
    tree_banknote_classifier.train(inputs_train, outputs_train)

    # Save it
    pk.save(tree_banknote_classifier, 'tree_banknote_classifier.pkl')

    # Print accuracy
    accuracy = tree_banknote_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = tree_banknote_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    tree_banknote_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['False', 'True'])

    # Plot descision tree
    tree_banknote_classifier.show_tree()

    # Assert accuracy
    assert (tree_banknote_classifier.accuracy(inputs_test, outputs_test)) >= 97

if __name__ == '__main__':
    try:
        test_banknote_tree.__wrapped__()
    except AssertionError:
        pass

