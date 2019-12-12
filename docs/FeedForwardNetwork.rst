Feed-Forward Neural Network
===========================

Class Reference
---------------

.. autoclass:: pykitml.NeuralNetwork
    
    .. automethod:: __init__

    .. automethod:: feed

    .. automethod:: get_output

    .. automethod:: get_output_onehot

    .. automethod:: train

    .. automethod:: plot_performance

    .. automethod:: cost

    .. automethod:: accuracy

    .. automethod:: confusion_matrix

    .. autoattribute:: nlayers

Example: Handwritten Digit Recognition (MNIST)
----------------------------------------------

.. literalinclude:: ../tests/test_mnist.py
   :pyobject: test_adam
   :lines: 2-
   :end-before: # Assert
   :dedent: 1