import sys
import os.path

import numpy as np
import pykitml as pk
from pykitml import mnist

def test_download():
    # Download the mnist data set
    mnist.get()
    # Test ran successfully
    assert True

def test_adagrad():

    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    digit_classifier = pk.NeuralNetwork([784, 100, 10])

    # Train it
    digit_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.Adagrad(learning_rate=0.07, decay_rate=0.99), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=10,
        decay_freq=10
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Test if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 94

def test_nesterov():

    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    digit_classifier = pk.NeuralNetwork([784, 100, 10])

    # Train it
    digit_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.Nesterov(learning_rate=0.1, decay_rate=0.99), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=10,
        decay_freq=10
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Test if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 94

def test_relu_nesterov():

    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    digit_classifier = pk.NeuralNetwork([784, 100, 10], config='relu-softmax-cross_entropy')

    # Train it
    digit_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.Nesterov(learning_rate=0.1, decay_rate=0.99), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=10,
        decay_freq=10
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Test if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 94

def test_momentum():
    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    digit_classifier = pk.NeuralNetwork([784, 100, 10])
    
    # Train it
    digit_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.Momentum(learning_rate=0.1, decay_rate=0.95), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=10,
        decay_freq=20
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')
    
    # Test if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 94

def test_gradient_descent():
    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    digit_classifier = pk.NeuralNetwork([784, 100, 10])
    
    # Train it
    digit_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.GradientDescent(learning_rate=0.2, decay_rate=0.99), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=10,
        decay_freq=20
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')
    
    # Test if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 92

def test_adam():
    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    digit_classifier = pk.NeuralNetwork([784, 100, 10])
    
    # Train it
    digit_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.Adam(learning_rate=0.012, decay_rate=0.95), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=10,
        decay_freq=15
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')
    
    # Test if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 95

def test_RMSprop():
    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()
    
    # Create a new neural network
    digit_classifier = pk.NeuralNetwork([784, 100, 10])
    
    # Train it
    digit_classifier.train(
        training_data=training_data,
        targets=training_targets, 
        batch_size=50, 
        epochs=1200, 
        optimizer=pk.RMSprop(learning_rate=0.012, decay_rate=0.95), 
        testing_data=testing_data, 
        testing_targets=testing_targets,
        testing_freq=10,
        decay_freq=15
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')
    
    # Test if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 95

if __name__ == '__main__':
    # List of optimizers
    optimizers = [
        'gradient_descent', 'momentum', 'nesterov',
        'adagrad', 'RMSprop', 'adam' 
    ]
    # Chack if arguments passed to the script is correct
    if(len(sys.argv) != 2 or sys.argv[1] not in optimizers):
        print('Usage: python3 test_mnist.py OPTIMIZER')
        print('List of available optimizers:')
        print(str(optimizers))
        exit()
    
    # If the dataset is not available then download it
    if(not os.path.exists('mnist.pkl')): mnist.get()

    # Run the requested optmizer test function
    try:
        locals()['test_'+sys.argv[1]]()
    except AssertionError:
        pass

    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()

    # Show performance
    digit_classifier = pk.load('digit_classifier_network.pkl')
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    digit_classifier.plot_performance()
