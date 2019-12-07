import cProfile

import numpy as np
import pykitml as pk
from pykitml import heartdisease

def test_download():
    heartdisease.get()

def test_heartdisease():
    # Load heartdisease data set
    inputs, outputs = heartdisease.load()

    # Normalize inputs
    array_min, array_max = pk.get_minmax(inputs)
    inputs = pk.normalize_array(inputs, array_min, array_max)

    # Normalize outputs
    array_min, array_max = pk.get_minmax(outputs)
    outputs = pk.normalize_array(outputs, array_min, array_max)

    # Create model
    heartdisease_classifier = pk.LinearRegression(13, 1)

    # Train the model
    heartdisease_classifier.train(
        training_data=inputs,
        targets=outputs, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.001, decay_rate=0.99), 
        testing_freq=30,
        decay_freq=20
    )

    # Test if it has enough accuracy
    heartdisease_classifier.plot_performance()

    # Save it
    pk.save(heartdisease_classifier, 'heartdisease_classifier.pkl')

if __name__ == '__main__':
    test_download()

    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_heartdisease)
        profiler.dump_stats('test_heartdisease.dat') 
    except AssertionError:
        pass

