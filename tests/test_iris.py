import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import iris

def test_disable_plot():
    # Disable plotting to prevent blocking tests
    pk._plotting._disable_ploting()

def test_iris():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs_train, outputs_train, inputs_test, outputs_test = iris.load()

    # Normalize inputs in the dataset
    inputs_min, inputs_max = pk.get_minmax(inputs_train)
    inputs_train = pk.normalize_minmax(inputs_train, inputs_min, inputs_max)
    inputs_test = pk.normalize_minmax(inputs_test, inputs_min, inputs_max)

    # Create model
    iris_classifier = pk.LogisticRegression(4, 3)

    # Train the model
    iris_classifier.train(
        training_data=inputs_train,
        targets=outputs_train, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.4, decay_rate=0.99), 
        testing_data=inputs_test,
        testing_targets=outputs_test,
        testing_freq=30,
        decay_freq=20
    )

    # Save it
    pk.save(iris_classifier, 'iris_classifier.pkl') 

    # Print accuracy
    accuracy = iris_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = iris_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot performance
    iris_classifier.plot_performance()

    # Plot confusion matrix
    iris_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert if it has enough accuracy
    assert iris_classifier.accuracy(inputs_train, outputs_train) >= 98

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris)
        profiler.dump_stats('test_iris.dat') 
    except AssertionError:
        pass