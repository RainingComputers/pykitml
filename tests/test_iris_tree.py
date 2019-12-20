import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import iris

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_iris_tree():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import iris

    # Load iris data set
    inputs, outputs = iris.load()

    # Create model
    tree_iris_classifier = pk.DecisionTree(4, 3, max_depth=4, feature_type=['continues']*4)

    # Train
    tree_iris_classifier.train(inputs, outputs)

    # Save it
    pk.save(tree_iris_classifier, 'tree_iris_classifier.pkl')

    # Print accuracy
    accuracy = tree_iris_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    tree_iris_classifier.confusion_matrix(inputs, outputs, 
        gnames=['Setosa', 'Versicolor', 'Virginica'])

    # Plot descision tree
    tree_iris_classifier.show_tree()

    # Assert accuracy
    assert (tree_iris_classifier.accuracy(inputs, outputs)) >= 98

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_iris_tree)
        profiler.dump_stats('test_iris_tree.dat') 
    except AssertionError:
        pass