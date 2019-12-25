# pykitml
Machine Learning library written in Python and NumPy.

Documentation: https://pykitml.readthedocs.io/en/latest/

# Demo (MNIST)
### Download MNIST:
```python
from pykitml.datasets import mnist

# Download the mnist data set
mnist.get()
```

### Training:
```python
import numpy as np
import pykitml as pk
from pykitml.datasets import mnist

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
print('Predicted: ', str(digit_classifier.get_output_onehot()))
```