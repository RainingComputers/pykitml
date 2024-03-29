from pykitml.testing import pktest_graph, pktest_nograph


@pktest_graph
def test_iris_bayes():
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs_train, outputs_train, inputs_test, outputs_test = iris.load()

    # Create model
    bayes_iris_classifier = pk.GaussianNaiveBayes(4, 3)

    # Train
    bayes_iris_classifier.train(inputs_train, outputs_train)

    # Save it
    pk.save(bayes_iris_classifier, 'bayes_iris_classifier.pkl')

    # Print accuracy
    accuracy = bayes_iris_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = bayes_iris_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    bayes_iris_classifier.confusion_matrix(inputs_test, outputs_test,
                                           gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert accuracy
    assert (bayes_iris_classifier.accuracy(inputs_train, outputs_train)) >= 95


@pktest_nograph
def test_predict_iris_bayes():
    import numpy as np
    import pykitml as pk

    # Predict type of species with
    # sepal-length sepal-width petal-length petal-width
    # 5.8, 2.7, 3.9, 1.2
    input_data = np.array([5.8, 2.7, 3.9, 1.2])

    # Load the model
    bayes_iris_classifier = pk.load('bayes_iris_classifier.pkl')

    # Get output
    bayes_iris_classifier.feed(input_data)
    model_output = bayes_iris_classifier.get_output_onehot()

    # Print result
    print(model_output)


if __name__ == '__main__':
    try:
        test_iris_bayes.__wrapped__()
        test_predict_iris_bayes.__wrapped__()
    except AssertionError:
        pass
