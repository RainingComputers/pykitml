from pykitml.testing import pktest_graph


@pktest_graph
def test_adult_forest():
    import os

    import pykitml as pk
    from pykitml.datasets import adult

    # Download the dataset
    if not os.path.exists('adult.data.pkl'):
        adult.get()

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
    forest_adult_classifier = pk.RandomForest(13, 2, max_depth=1000, feature_type=ftypes)

    # Train
    forest_adult_classifier.train(inputs_train, outputs_train, num_trees=100)

    # Save it
    pk.save(forest_adult_classifier, 'forest_adult_classifier.pkl')

    # Print accuracy
    accuracy = forest_adult_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = forest_adult_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    forest_adult_classifier.confusion_matrix(inputs_test, outputs_test,
                                             gnames=['False', 'True'])

    # Assert accuracy
    assert (forest_adult_classifier.accuracy(inputs_test, outputs_test)) >= 82


if __name__ == '__main__':
    try:
        test_adult_forest.__wrapped__()
    except AssertionError:
        pass
