from pykitml.testing import pktest_graph, pktest_nograph

import pytest

@pktest_graph
def test_sonar_forest():
    import os

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import sonar

    # Download the dataset
    if(not os.path.exists('sonar.pkl')): sonar.get()

    # Load the sonar dataset
    inputs_train, outputs_train, inputs_test, outputs_test = sonar.load()
    outputs_train = pk.onehot(outputs_train)
    outputs_test = pk.onehot(outputs_test)

    # Create model
    forest_sonar_classifier = pk.RandomForest(60, 2, max_depth=9, feature_type=['continues']*60)

    # Train the model
    forest_sonar_classifier.train(inputs_train, outputs_train, num_feature_bag=60)

    # Save it
    pk.save(forest_sonar_classifier, 'forest_sonar_classifier.pkl')

    # Print accuracy
    accuracy = forest_sonar_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = forest_sonar_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    forest_sonar_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['False', 'True'])

if __name__ == '__main__':
    try:
        test_sonar_forest.__wrapped__()
    except AssertionError:
        pass