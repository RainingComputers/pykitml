import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import banknote

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_download():
    banknote.get()

def test_banknote():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import banknote

    # Download the dataset 
    if(not os.path.exists('banknote.pkl')): banknote.get()

    # Load banknote data set
    inputs_train, outputs_train, inputs_test, outputs_test = banknote.load()

    # Normalize dataset
    array_min, array_max = pk.get_minmax(inputs_train)
    inputs_train = pk.normalize_minmax(inputs_train, array_min, array_max)
    inputs_test = pk.normalize_minmax(inputs_test, array_min, array_max)

    # Create model
    banknote_classifier = pk.LogisticRegression(4, 1)

    # Train the model
    banknote_classifier.train(
        training_data=inputs_train,
        targets=outputs_train, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.06, decay_rate=0.99),
        testing_data=inputs_test,
        testing_targets=outputs_test, 
        testing_freq=30,
        decay_freq=40
    )

    # Save it
    pk.save(banknote_classifier, 'banknote_classifier.pkl') 

    # Plot performance
    banknote_classifier.plot_performance()
    
    # Print accuracy
    accuracy = banknote_classifier.accuracy(inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = banknote_classifier.accuracy(inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot confusion matrix
    banknote_classifier.confusion_matrix(inputs_test, outputs_test)

    # Assert if it has enough accuracy
    assert banknote_classifier.accuracy(inputs_test, outputs_test) >= 82

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_banknote)
        profiler.dump_stats('test_banknote.dat') 
    except AssertionError:
        pass