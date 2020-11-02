from pykitml.testing import pktest_graph, pktest_nograph

@pktest_nograph
def test_search():
    import os

    import numpy as np
    import pykitml as pk
    from pykitml.datasets import mnist

    # If the dataset is not available then download it
    if(not os.path.exists('mnist.pkl')): mnist.get(type='fashion')

    # Load dataset
    training_data, training_targets, testing_data, testing_targets = mnist.load()

    # Search for hyperparameters
    #   Learning rate alpha = 10^-4 to 10^-2
    #   Decay rate = 0.8 to 1
    #   Decay frequency = 10 to 30
    #   Batch size = 10 to 100
    search = pk.RandomSearch()
    for alpha, decay, decay_freq, bsize in search.search(10, 3, 5,
        [-4, -2, 'log'], [0.8, 1, 'float'], [10, 30, 'int'], [10, 100, 'int']):
          

        # Create a new neural network
        fashion_classifier = pk.NeuralNetwork([784, 100, 10])

        # Train it
        fashion_classifier.train(
            training_data=training_data,
            targets=training_targets, 
            batch_size=bsize, 
            epochs=1200, 
            optimizer=pk.Adam(learning_rate=alpha, decay_rate=decay), 
            testing_freq=100,
            decay_freq=decay_freq
        )

        cost = fashion_classifier.cost(testing_data, testing_targets)
        search.set_cost(cost)

        # Save the best model
        if(search.best): pk.save(fashion_classifier, 'best.pkl')

    # Load the best model
    fashion_classifier = pk.load('best.pkl')

    # Show performance       
    accuracy = fashion_classifier.accuracy(testing_data, testing_targets)
    print('Test Accuracy:', accuracy)

    # Assert accuracy
    assert accuracy > 84

if __name__ == '__main__':
    try:
        test_search.__wrapped__()
    except AssertionError:
        pass