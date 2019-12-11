import sys
import os.path
import cProfile

import numpy as np
import pykitml as pk
from pykitml import mnist

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._base._disable_ploting()

def test_download():
    # Download the mnist data set
    mnist.get(type='fashion')
    # Test ran successfully
    assert True

def test_adam():
    # If the dataset is not available then download it
    if(not os.path.exists('mnist.pkl')): mnist.get(type='fashion')

    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    fashion_classifier = pk.NeuralNetwork([784, 100, 10])
    
    # Train it
    fashion_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.Adam(learning_rate=0.012, decay_rate=0.95), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=30,
        decay_freq=10
    )
    
    # Save it
    pk.save(fashion_classifier, 'fashion_classifier_network.pkl')

    # Show performance
    fashion_classifier = pk.load('fashion_classifier_network.pkl')
    accuracy = fashion_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = fashion_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)

    # Plot performance
    fashion_classifier.plot_performance()

    # Show confusion matrix
    fashion_classifier.confusion_matrix(
        training_data, training_targets,
        gnames = ['T-shirt/Top', 'Trouser', 'Pullover',
                'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                'Bag', 'Ankle Boot'
            ]
    )

    # Assert if it has enough accuracy
    assert fashion_classifier.accuracy(training_data, training_targets) > 84

if __name__ == '__main__':
    # Run the requested optmizer test function
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_adam)
        profiler.dump_stats('test_mnist_fasion.dat') 
    except AssertionError:
        pass
    