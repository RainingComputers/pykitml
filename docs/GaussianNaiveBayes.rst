Gaussian Naive Bayes
====================

Class Reference
---------------

.. autoclass:: pykitml.GaussianNaiveBayes
    
    .. automethod:: __init__

    .. automethod:: feed

    .. automethod:: get_output

    .. automethod:: get_output_one_hot

    .. automethod:: train

    .. automethod:: accuracy

    .. automethod:: confusion_matrix

Example: Classifying Iris
-------------------------

.. literalinclude:: ../tests/test_iris_bayes.py
   :pyobject: test_iris_bayes
   :lines: 2-
   :end-before: # Assert
   :dedent: 1