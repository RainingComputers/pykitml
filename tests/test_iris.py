import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import iris

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_iris():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs, outputs = iris.load()

    # Normalize inputs in the dataset
    inputs_min, inputs_max = pk.get_minmax(inputs)
    inputs = pk.normalize_minmax(inputs, inputs_min, inputs_max)

    # Create model
    iris_classifier = pk.LogisticRegression(4, 3)

    # Train the model
    iris_classifier.train(
        training_data=inputs,
        targets=outputs, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.4, decay_rate=0.99), 
        testing_freq=30,
        decay_freq=20
    )

    # Save it
    pk.save(iris_classifier, 'iris_classifier.pkl') 

    # Print accuracy and plot performance
    iris_classifier.plot_performance()
    accuracy = iris_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    iris_classifier.confusion_matrix(inputs, outputs, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert if it has enough accuracy
    assert iris_classifier.accuracy(inputs, outputs) >= 98

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris)
        profiler.dump_stats('test_iris.dat') 
    except AssertionError:
        pass