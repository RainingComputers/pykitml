import sys
import os.path
import cProfile

import numpy as np
import pykitml as pk
from pykitml.datasets import mnist

import pytest

def test_disable_plot():
    # Diable plotting to prevent blocking tests
    pk._plotting._disable_ploting()

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
        testing_freq=30,
        decay_freq=10
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Show performance
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    
    # Plot performance graph
    digit_classifier.plot_performance()

    # Show confusion matrix
    digit_classifier.confusion_matrix(training_data, training_targets)

    # Assert if it has enough accuracy
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
        testing_freq=30,
        decay_freq=10
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Show performance
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    
    # Plot performance graph
    digit_classifier.plot_performance()

    # Show confusion matrix
    digit_classifier.confusion_matrix(training_data, training_targets)

    # Assert if it has enough accuracy
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
        testing_freq=30,
        decay_freq=10
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Show performance
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    
    # Plot performance graph
    digit_classifier.plot_performance()

    # Show confusion matrix
    digit_classifier.confusion_matrix(training_data, training_targets)

    # Assert if it has enough accuracy
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
        testing_freq=30,
        decay_freq=20
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Show performance
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    
    # Plot performance graph
    digit_classifier.plot_performance()

    # Show confusion matrix
    digit_classifier.confusion_matrix(training_data, training_targets)

    # Assert if it has enough accuracy
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
        testing_freq=30,
        decay_freq=20
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Show performance
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    
    # Plot performance graph
    digit_classifier.plot_performance()

    # Show confusion matrix
    digit_classifier.confusion_matrix(training_data, training_targets)

    # Assert if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 92

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
        testing_freq=30,
        decay_freq=15
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Show performance
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    
    # Plot performance graph
    digit_classifier.plot_performance()

    # Show confusion matrix
    digit_classifier.confusion_matrix(training_data, training_targets)

    # Assert if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 95

def test_adam():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import mnist
    
    # Download dataset
    if(not os.path.exists('mnist.pkl')): mnist.get()

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
        testing_freq=30,
        decay_freq=15
    )
    
    # Save it
    pk.save(digit_classifier, 'digit_classifier_network.pkl')

    # Show performance
    accuracy = digit_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)        
    accuracy = digit_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)
    
    # Plot performance graph
    digit_classifier.plot_performance()

    # Show confusion matrix
    digit_classifier.confusion_matrix(training_data, training_targets)

    # Assert if it has enough accuracy
    assert digit_classifier.accuracy(training_data, training_targets) > 95

@pytest.mark.skip(reason='Will block other tests')
def test_predict():
    import random

    import numpy as np
    import matplotlib.pyplot as plt
    import pykitml as pk
    from pykitml.datasets import mnist

    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()

    # Load the trained network
    digit_classifier = pk.load('digit_classifier_network.pkl')

    # Pick a random example from testing data
    index = random.randint(0, 9999)

    # Show the test data and the label
    plt.imshow(training_data[index].reshape(28, 28))
    plt.show()
    print('Label: ', training_targets[index])

    # Show prediction
    digit_classifier.feed(training_data[index])
    model_output = digit_classifier.get_output_onehot()
    print('Predicted: ', model_output)


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
        profiler = cProfile.Profile()
        profiler.runcall(locals()['test_'+sys.argv[1]])
        profiler.dump_stats('test_mnist_'+sys.argv[1]+'.dat') 

        test_predict()
    except AssertionError:
        pass
