import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import iris

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_iris_svm():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs_train, outputs_train, inputs_test, outputs_test = iris.load()

    # Format the outputs for svm training, zeros to -1
    svm_outputs_train = np.where(outputs_train==0, -1, 1)
    svm_outputs_test = np.where(outputs_test==0, -1, 1)

    # Create model
    svm_iris_classifier = pk.LinearSVM(4, 3)

    # Train the model
    svm_iris_classifier.train(
        training_data=inputs_train,
        targets=svm_outputs_train, 
        batch_size=20, 
        epochs=1000, 
        optimizer=pk.Adam(learning_rate=3, decay_rate=0.95),
        testing_data=inputs_test,
        testing_targets=svm_outputs_test, 
        testing_freq=30,
        decay_freq=10
    )

    # Save it
    pk.save(svm_iris_classifier, 'svm_iris_classifier.pkl')

    # Print accuracy
    accuracy = svm_iris_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = svm_iris_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot performance
    svm_iris_classifier.plot_performance()

    # Plot confusion matrix
    svm_iris_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert if it has enough accuracy
    assert svm_iris_classifier.accuracy(inputs_train, outputs_train) >= 97

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris_svm)
        profiler.dump_stats('test_iris_svm.dat') 
    except AssertionError:
        pass