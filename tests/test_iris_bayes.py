import cProfile

import numpy as np
import pykitml as pk
from pykitml import iris

def test_iris_bayes():
    # Load iris data set
    inputs, outputs = iris.load()

    # Create model
    bayes_iris_classifier = pk.NaiveBayes(4, 3)

    # Train
    bayes_iris_classifier.train(inputs, outputs)

    # Check accuracy
    assert (bayes_iris_classifier.accuracy(inputs, outputs)) >= 96

    # Save it
    pk.save(bayes_iris_classifier, 'bayes_iris_classifier.pkl')

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris_bayes)
        profiler.dump_stats('test_iris_bayes.dat') 
    except AssertionError:
        pass

    # Load dataset
    inputs, outputs = iris.load()

    # Load model
    bayes_iris_classifier = pk.load('bayes_iris_classifier.pkl')

    # Print accuracy
    accuracy = bayes_iris_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    bayes_iris_classifier.confusion_matrix(inputs, outputs, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])