import os
import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import banknote

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_banknote_forest():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import banknote

    # Download the dataset 
    if(not os.path.exists('banknote.pkl')): banknote.get()

    # Load heart data set
    inputs_train, outputs_train, inputs_test, outputs_test = banknote.load()
    
    # Change 0/False to [1, 0]
    # Change 1/True to [0, 1]
    outputs_train = pk.onehot(outputs_train)
    outputs_test = pk.onehot(outputs_test)

    # Create model
    ftypes = ['continues']*4
    forest_banknote_classifier = pk.RandomForest(4, 2, max_depth=9, feature_type=ftypes)

    # Train
    forest_banknote_classifier.train(inputs_train, outputs_train)

    # Save it
    pk.save(forest_banknote_classifier, 'forest_banknote_classifier.pkl')

    # Print accuracy
    accuracy = forest_banknote_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = forest_banknote_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    forest_banknote_classifier.confusion_matrix(inputs_test, outputs_test, 
        gnames=['False', 'True'])

    # Assert accuracy
    assert (forest_banknote_classifier.accuracy(inputs_test, outputs_test)) >= 98

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_banknote_forest)
        profiler.dump_stats('test_banknote_forest.dat') 
    except AssertionError:
        pass

