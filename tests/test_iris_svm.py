import cProfile

import numpy as np
import pykitml as pk
from pykitml import iris

def test_iris_svm():
    # Load iris data set
    inputs, outputs = iris.load()

    # Format the outputs for svm training, zeroes to -1
    outputs_train = np.where(outputs==0, -1, 1)

    # Create model
    svm_iris_classifier = pk.LinearSVM(4, 3)

    # Train the model
    svm_iris_classifier.train(
        training_data=inputs,
        targets=outputs_train, 
        batch_size=20, 
        epochs=1000, 
        optimizer=pk.Adam(learning_rate=3, decay_rate=0.95), 
        testing_freq=30,
        decay_freq=10
    )

    # Test if it has enough accuracy
    assert svm_iris_classifier.accuracy(inputs, outputs) >= 98

    # Save it
    pk.save(svm_iris_classifier, 'svm_iris_classifier.pkl')

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris_svm)
        profiler.dump_stats('test_iris_svm.dat') 
    except AssertionError:
        pass

    # Load dataset
    inputs, outputs = iris.load()

    # Load model
    svm_iris_classifier = pk.load('svm_iris_classifier.pkl')

    # Print accuracy and plor performance
    svm_iris_classifier.plot_performance()
    accuracy = svm_iris_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    svm_iris_classifier.confusion_matrix(inputs, outputs, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])