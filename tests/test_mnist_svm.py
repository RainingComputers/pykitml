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

def test_mnist_svm():
    import os.path

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import mnist
    
    # Download dataset
    if(not os.path.exists('mnist.pkl')): mnist.get()

    # Load mnist data set
    inputs_train, outputs_train, inputs_test, outputs_test = mnist.load()

    # Train on only first 10000
    inputs_train = inputs_train[:10000]
    outputs_train = outputs_train[:10000]

    # Transform inputs using gaussian kernal
    sigma = 3.15
    gaussian_inputs_train = pk.gaussian_kernel(inputs_train, inputs_train, sigma)
    gaussian_inputs_test = pk.gaussian_kernel(inputs_test, inputs_train, sigma)

    # Format the outputs for svm training, zeros to -1
    svm_outputs_train = np.where(outputs_train==0, -1, 1)
    svm_outputs_test = np.where(outputs_test==0, -1, 1)

    # Create model
    svm_mnist_classifier = pk.SVM(gaussian_inputs_train.shape[1], 10)

    # Train the model
    svm_mnist_classifier.train(
        training_data=gaussian_inputs_train,
        targets=svm_outputs_train, 
        batch_size=20, 
        epochs=1000, 
        optimizer=pk.Adam(learning_rate=3.5, decay_rate=0.95),
        testing_data=gaussian_inputs_test,
        testing_targets=svm_outputs_test, 
        testing_freq=30,
        decay_freq=10
    )

    # Save it
    pk.save(svm_mnist_classifier, 'svm_mnist_classifier.pkl')

    # Print accuracy
    accuracy = svm_mnist_classifier.accuracy(gaussian_inputs_train, outputs_train)
    print('Train accuracy:', accuracy)
    accuracy = svm_mnist_classifier.accuracy(gaussian_inputs_test, outputs_test)
    print('Test accuracy:', accuracy)

    # Plot performance
    svm_mnist_classifier.plot_performance()

    # Plot confusion matrix
    svm_mnist_classifier.confusion_matrix(gaussian_inputs_test, outputs_test)

    # Assert if it has enough accuracy
    assert svm_mnist_classifier.accuracy(gaussian_inputs_train, outputs_train) >= 90

@pytest.mark.skip(reason='Will block other tests')
def test_predict():
    import random

    import numpy as np
    import matplotlib.pyplot as plt
    import pykitml as pk
    from pykitml.datasets import mnist

    # Load dataset
    inputs_train, outputs_train, inputs_test, outputs_test = mnist.load()

    # Use only first 10000
    inputs_train = inputs_train[:10000]
    outputs_train = outputs_train[:10000]

    # Load the trained network
    svm_mnist_classifier = pk.load('svm_mnist_classifier.pkl')

    # Pick a random example from testing data
    index = random.randint(0, 9000)

    # Show the test data and the label
    plt.imshow(inputs_train[index].reshape(28, 28))
    plt.show()
    print('Label: ', outputs_train[index])

    # Transform the input
    input_data = pk.gaussian_kernel(inputs_train[index], inputs_train)

    # Show prediction
    svm_mnist_classifier.feed(input_data)
    model_output = svm_mnist_classifier.get_output_onehot()
    print('Predicted: ', model_output)


if __name__ == '__main__':
    try:
        profiler = cProfile.Profile()
        profiler.runcall(test_mnist_svm)
        profiler.dump_stats('test_mnist_svm.dat') 

        test_predict()
    except AssertionError:
        pass