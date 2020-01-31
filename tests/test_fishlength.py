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

    # Create plynomial features
    inputs_poly = pk.polynomial(inputs)

    # Normalize outputs
    array_min, array_max = pk.get_minmax(outputs)
    outputs = pk.normalize_minmax(outputs, array_min, array_max)

    # Create model
    fish_classifier = pk.LinearRegression(inputs_poly.shape[1], 1)

    # Train the model
    fish_classifier.train(
        training_data=inputs_poly,
        targets=outputs, 
        batch_size=22, 
        epochs=1000, 
        optimizer=pk.Adam(learning_rate=0.02, decay_rate=0.99), 
        testing_freq=1,
        decay_freq=10
    )

    # Save model
    pk.save(fish_classifier, 'fish_classifier.pkl')

    # Plot performance
    fish_classifier.plot_performance()

    # Assert if it has enough accuracy
    assert fish_classifier.cost(inputs_poly, outputs) <= 0

def test_predict():
    import numpy as np
    import pykitml as pk
    from pykitml.datasets import fishlength

    # Predict length of fish that is 28 days old at 25C

    # Load the dataset
    inputs, outputs = fishlength.load()

    # Load the model
    fish_classifier = pk.load('fish_classifier.pkl')
    
    # Normalize inputs
    array_min, array_max = pk.get_minmax(inputs)
    input_data = pk.normalize_minmax(np.array([28, 25]), array_min, array_max)

    # Create plynomial features
    input_data_poly = pk.polynomial(input_data)
    
    # Get output
    fish_classifier.feed(input_data_poly)
    model_output = fish_classifier.get_output()

    # Denormalize output
    array_min, array_max = pk.get_minmax(outputs)
    model_output = pk.denormalize_minmax(model_output, array_min, array_max)

    # Print result
    print(model_output)


if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_fishlength)
        profiler.dump_stats('test_fishlength.dat')

        test_predict() 
    except AssertionError:
        pass

