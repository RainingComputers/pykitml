from pykitml.testing import pktest_graph, pktest_nograph


@pktest_graph
def test_iris_tree():
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs_train, outputs_train, inputs_test, outputs_test = iris.load()

    # Create model
    tree_iris_classifier = pk.DecisionTree(4, 3, max_depth=5, feature_type=['continues']*4)

    # Train
    tree_iris_classifier.train(inputs_train, outputs_train)

    # Save it
    pk.save(tree_iris_classifier, 'tree_iris_classifier.pkl')

    # Print accuracy
    accuracy = tree_iris_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = tree_iris_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    tree_iris_classifier.confusion_matrix(inputs_test, outputs_test,
                                          gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Plot decision tree
    tree_iris_classifier.show_tree()

    # Assert accuracy
    assert (tree_iris_classifier.accuracy(inputs_train, outputs_train)) >= 98


@pktest_nograph
def test_predict_iris_tree():
    import numpy as np
    import pykitml as pk

    # Predict type of species with
    # sepal-length sepal-width petal-length petal-width
    # 5.8, 2.7, 3.9, 1.2
    input_data = np.array([5.8, 2.7, 3.9, 1.2])

    # Load the model
    tree_iris_classifier = pk.load('tree_iris_classifier.pkl')

    # Get output
    tree_iris_classifier.feed(input_data)
    model_output = tree_iris_classifier.get_output_onehot()

    # Print result
    print(model_output)


if __name__ == '__main__':
    try:
        test_iris_tree.__wrapped__()
        test_predict_iris_tree.__wrapped__()
    except AssertionError:
        pass
