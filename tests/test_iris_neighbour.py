import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import iris

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_iris_neighbour():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs_train, outputs_train, inputs_test, outputs_test = iris.load()

    # Create model
    neighbour_iris_classifier = pk.NearestNeighbour(4, 3)

    # Train the model
    neighbour_iris_classifier.train(
        training_data=inputs_train,
        targets=outputs_train, 
    )

    # Save it
    pk.save(neighbour_iris_classifier, 'neighbour_iris_classifier.pkl') 

    # Print accuracy
    accuracy = neighbour_iris_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = neighbour_iris_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    neighbour_iris_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert if it has enough accuracy
    assert neighbour_iris_classifier.accuracy(inputs_train, outputs_train) >= 100

def test_predict():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Predict type of species with 
    # sepal-length sepal-width petal-length petal-width
    # 5.8, 2.7, 3.9, 1.2
    input_data = np.array([5.8, 2.7, 3.9, 1.2])

    # Load the model
    neighbour_iris_classifier = pk.load('neighbour_iris_classifier.pkl')

    # Get output
    neighbour_iris_classifier.feed(input_data)
    model_output = neighbour_iris_classifier.get_output_onehot()

    # Print result
    print(model_output)

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris_neighbour)
        profiler.dump_stats('test_iris_neighbour.dat') 

        test_predict()
    except AssertionError:
        pass