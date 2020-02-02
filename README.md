![pykitml logo](pykitml128.png)

# pykitml (Python Kit for Machine Learning)
Machine Learning library written in Python and NumPy.

Documentation: https://pykitml.readthedocs.io/en/latest/

# Demo (MNIST)
### Training:
``` python
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
```

### Trying the model:
```python
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
```

### Performance Graph

![Performance Graph](docs/demo_pics/neural_network_perf_graph.png)

## Confusion Matrix

![Confusion Matrix](docs/demo_pics/neural_network_confusion_matrix.png)

