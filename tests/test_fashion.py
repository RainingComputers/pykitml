from pykitml.testing import pktest_graph


def test_download():
    from pykitml.datasets import mnist
    # Download the mnist data set
    mnist.get(type='fashion')
    # Test ran successfully
    assert True


@pktest_graph
def test_adam_fashion():
    import os

    import pykitml as pk
    from pykitml.datasets import mnist

    # If the dataset is not available then download it
    if not os.path.exists('mnist.pkl'):
        mnist.get(type='fashion')

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
    accuracy = fashion_classifier.accuracy(training_data, training_targets)
    print('Train Accuracy:', accuracy)
    accuracy = fashion_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)

    # Plot performance
    fashion_classifier.plot_performance()

    # Show confusion matrix
    fashion_classifier.confusion_matrix(
        training_data, training_targets,
        gnames=['T-shirt/Top', 'Trouser', 'Pullover',
                'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker',
                'Bag', 'Ankle Boot'
                ]
    )

    # Assert if it has enough accuracy
    assert fashion_classifier.accuracy(training_data, training_targets) > 84


if __name__ == '__main__':
    try:
        test_adam_fashion.__wrapped__()
    except AssertionError:
        pass
