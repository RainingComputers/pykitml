import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import iris

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_iris_bayes():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs, outputs = iris.load()

    # Create model
    bayes_iris_classifier = pk.GaussianNaiveBayes(4, 3)

    # Train
    bayes_iris_classifier.train(inputs, outputs)

    # Save it
    pk.save(bayes_iris_classifier, 'bayes_iris_classifier.pkl')

    # Print accuracy
    accuracy = bayes_iris_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    bayes_iris_classifier.confusion_matrix(inputs, outputs, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Assert accuracy
    assert (bayes_iris_classifier.accuracy(inputs, outputs)) >= 96

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris_bayes)
        profiler.dump_stats('test_iris_bayes.dat') 
    except AssertionError:
        pass
