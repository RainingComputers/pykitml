import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import fishlength

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_fishlength():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import fishlength

    # Load the dataset
    inputs, outputs = fishlength.load()

    # Normalize inputs
    array_min, array_max = pk.get_minmax(inputs)
    inputs = pk.normalize_minmax(inputs, array_min, array_max)

    # Normalize outputs
    array_min, array_max = pk.get_minmax(outputs)
    outputs = pk.normalize_minmax(outputs, array_min, array_max)

    # Create model
    fish_classifier = pk.LinearRegression(2, 1)

    # Train the model
    fish_classifier.train(
        training_data=inputs,
        targets=outputs, 
        batch_size=22, 
        epochs=100, 
        optimizer=pk.Adam(learning_rate=0.02, decay_rate=0.99), 
        testing_freq=1,
        decay_freq=10
    )

    # Save model
    pk.save(fish_classifier, 'fish_classifier.pkl')

    # Plot performance
    fish_classifier.plot_performance()

    # Assert if it has enough accuracy
    assert fish_classifier.cost(inputs, outputs) <= 10

if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_fishlength)
        profiler.dump_stats('test_fishlength.dat') 
    except AssertionError:
        pass

