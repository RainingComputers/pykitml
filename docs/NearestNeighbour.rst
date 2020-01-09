Nearest Neighbour
=================

Class Reference
---------------

.. autoclass:: pykitml.NearestNeighbour
    
    .. automethod:: __init__

    .. automethod:: feed

    .. automethod:: get_output

    .. automethod:: get_output_onehot

    .. automethod:: train

    .. automethod:: accuracy

    .. automethod:: confusion_matrix

Example: Classifying Iris
-------------------------

.. literalinclude:: ../tests/test_iris_neighbour.py
   :pyobject: test_iris_neighbour
   :lines: 2-
   :end-before: # Assert
   :dedent: 1