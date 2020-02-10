from pykitml.testing import pktest_graph, pktest_nograph

@pktest_graph
def test_iris_neighbor():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs_train, outputs_train, inputs_test, outputs_test = iris.load()

    # Create model
    neighbor_iris_classifier = pk.NearestNeighbor(4, 3)

    # Train the model
    neighbor_iris_classifier.train(
        training_data=inputs_train,
        targets=outputs_train, 
    )

    # Save it
    pk.save(neighbor_iris_classifier, 'neighbor_iris_classifier.pkl') 

    # Print accuracy
    accuracy = neighbor_iris_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = neighbor_iris_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    neighbor_iris_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert if it has enough accuracy
    assert neighbor_iris_classifier.accuracy(inputs_train, outputs_train) >= 100

@pktest_nograph
def test_predict_iris_neighbor():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Predict type of species with 
    # sepal-length sepal-width petal-length petal-width
    # 5.8, 2.7, 3.9, 1.2
    input_data = np.array([5.8, 2.7, 3.9, 1.2])

    # Load the model
    neighbor_iris_classifier = pk.load('neighbor_iris_classifier.pkl')

    # Get output
    neighbor_iris_classifier.feed(input_data)
    model_output = neighbor_iris_classifier.get_output_onehot()

    # Print result
    print(model_output)

if __name__ == '__main__':
    try:
        test_iris_neighbor.__wrapped__()
        test_predict_iris_neighbor.__wrapped__()
    except AssertionError:
        pass