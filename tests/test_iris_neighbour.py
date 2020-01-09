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
    inputs, outputs = iris.load()

    # Create model
    neighbour_iris_classifier = pk.NearestNeighbour(4, 3)

    # Train the model
    neighbour_iris_classifier.train(
        training_data=inputs,
        targets=outputs, 
    )

    # Save it
    pk.save(neighbour_iris_classifier, 'neighbour_iris_classifier.pkl') 

    # Print accuracy
    accuracy = neighbour_iris_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    neighbour_iris_classifier.confusion_matrix(inputs, outputs, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert if it has enough accuracy
    assert neighbour_iris_classifier.accuracy(inputs, outputs) >= 100

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris_neighbour)
        profiler.dump_stats('test_iris_neighbour.dat') 
    except AssertionError:
        pass