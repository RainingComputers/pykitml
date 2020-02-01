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

**Dataset**

:ref:`iris_dataset`

**Training**

.. literalinclude:: ../tests/test_iris_tree.py
   :pyobject: test_iris_tree
   :lines: 2-
   :end-before: # Assert
   :dedent: 1

**Predict type of species with sepal-length, sepal-width, petal-length, petal-width: 
5.8, 2.7, 3.9, 1.2**

.. literalinclude:: ../tests/test_iris_tree.py
   :pyobject: test_predict
   :lines: 2-
   :dedent: 1

**Tree Graph**

.. image:: ./demo_pics/tree.png

**Confusion Matrix**

.. image:: ./demo_pics/tree_confusion_matrix.png