import os.path

import numpy as np
from pykitml import diabetes
import pykitml as pk

def test_download():
    # Download the dataset
    diabetes.get()
    # Test ran successfully
    assert True

def test_diabetes():
    # Load dataset
    inputs, outputs = diabetes.load()
    
    # Normalize inputs in the dataset
    inputs_min, inputs_max = pk.get_minmax(inputs)
    inputs = pk.normalize_array(inputs, inputs_min, inputs_max)
    
    # Normalize outputs in the dataset
    outputs_min, outputs_max = pk.get_minmax(outputs)
    outputs = pk.normalize_array(outputs, outputs_min, outputs_max)
    
    # Split dataset into training and testing
    training_input = inputs[0:392]
    training_output = outputs[0:392]
    testing_input = inputs[392:441]
    testing_output = outputs[392:441]
    
    # Create a new neural network
    diabetes_network = pk.NeuralNetwork(
        layer_sizes=[10, 100, 1], 
        config='tanh-tanh-mse',
    )
    
    # Train it
    diabetes_network.train(
        training_data=training_input,
        targets=training_output,
        batch_size=1, 
        epochs=300,
        optimizer=pk.Nesterov(0.0007, 0.7),
        testing_data=testing_input,
        testing_targets=testing_output
    )
    
    # Save it
    pk.save(diabetes_network, 'diabetes_network.pkl')
    
    # Test ran successfully
    assert True

if __name__ == '__main__':
    if(not os.path.exists('diabetes.pkl')): diabetes.get()
    test_diabetes()    