import numpy as np
import pykitml as pk
from pykitml import iris

def test_iris():
    # Load iris data set
    inputs, outputs = iris.load()

    # Create model
    iris_classifier = pk.LogisticRegression(4, 3)

    # Train the model
    iris_classifier.train(
        training_data=inputs,
        targets=outputs, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.1, decay_rate=0.99), 
        testing_freq=30,
        decay_freq=20
    )

    # Test if it has enough accuracy
    assert iris_classifier.accuracy(inputs, outputs) >= 98

    # Save it
    pk.save(iris_classifier, 'iris_classifier.pkl')

def test_iris_normalization():
    # Load iris data set
    inputs, outputs = iris.load()

    # Normalize inputs in the dataset
    inputs_min, inputs_max = pk.get_minmax(inputs)
    inputs = pk.normalize_array(inputs, inputs_min, inputs_max)
    
    # Normalize outputs in the dataset
    outputs_min, outputs_max = pk.get_minmax(outputs)
    outputs = pk.normalize_array(outputs, outputs_min, outputs_max)

    # Create model
    iris_classifier = pk.LogisticRegression(4, 3)

    # Train the model
    iris_classifier.train(
        training_data=inputs,
        targets=outputs, 
        batch_size=10, 
        epochs=1500, 
        optimizer=pk.Adam(learning_rate=0.4, decay_rate=0.99), 
        testing_freq=30,
        decay_freq=20
    )

    # Test if it has enough accuracy
    assert iris_classifier.accuracy(inputs, outputs) >= 98

    # Save it
    pk.save(iris_classifier, 'iris_classifier.pkl') 


if __name__ == '__main__':
    test_iris_normalization()