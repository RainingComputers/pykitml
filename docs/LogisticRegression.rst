Logistic Regression
===================

Class Reference
---------------

.. autoclass:: pykitml.LogisticRegression
    
    .. automethod:: __init__

    .. automethod:: feed

    .. automethod:: get_output

    .. automethod:: get_output_onehot

    .. automethod:: train

    .. automethod:: plot_performance

    .. automethod:: cost

    .. automethod:: accuracy

    .. automethod:: confusion_matrix

Example: Classifying Iris
-------------------------

.. literalinclude:: ../tests/test_iris.py
   :pyobject: test_iris
   :lines: 2-
   :end-before: # Assert
   :dedent: 1

