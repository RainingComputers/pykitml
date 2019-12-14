import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import heartdisease

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_heart():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import heartdisease

    # Download the dataset 
    if(not os.path.exists('heartdisease.pkl')): heartdisease.get()

    # Load heartdisease data set
    inputs, outputs = heartdisease.load()

    # Normalize inputs in the dataset
    inputs_min, inputs_max = pk.get_minmax(inputs)
    inputs = pk.normalize_minmax(inputs, inputs_min, inputs_max, cols=[0, 3, 4, 7, 9])  

    # Change categorical values to onehot values
    inputs = pk.onehot_cols(inputs, [1, 2, 5, 6, 8, 10, 11, 12])      

    # Create model
    heart_classifier = pk.LogisticRegression(35, 1)

    # Train the model
    heart_classifier.train(
        training_data=inputs,
        targets=outputs, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.015, decay_rate=0.99), 
        testing_freq=30,
        decay_freq=40
    )

    # Save it
    pk.save(heart_classifier, 'heart_classifier.pkl') 

    # Print accuracy and plot performance
    heart_classifier.plot_performance()
    accuracy = heart_classifier.accuracy(inputs, outputs)
    print('Accuracy:', accuracy)

    # Plot confusion matrix
    heart_classifier.confusion_matrix(inputs, outputs)

    # Assert if it has enough accuracy
    assert heart_classifier.accuracy(inputs, outputs) >= 87

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_heart)
        profiler.dump_stats('test_heart.dat') 
    except AssertionError:
        pass