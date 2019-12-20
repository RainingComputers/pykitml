Decision Tree
=============

Class Reference
---------------

.. autoclass:: pykitml.DecisionTree
    
    .. automethod:: __init__

    .. automethod:: feed

    .. automethod:: get_output

    .. automethod:: get_output_onehot

    .. automethod:: train

    .. automethod:: accuracy

    .. automethod:: confusion_matrix

    .. automethod:: show_tree

Example: Classifying Iris
-------------------------

.. literalinclude:: ../tests/test_iris_tree.py
   :pyobject: test_iris_tree
   :lines: 2-
   :end-before: # Assert
   :dedent: 1