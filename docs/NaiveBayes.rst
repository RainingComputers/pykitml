Naive Bayes
===========

Class Reference
---------------

.. autoclass:: pykitml.NaiveBayes
    
    .. automethod:: __init__

    .. automethod:: feed

    .. automethod:: get_output

    .. automethod:: get_output_onehot

    .. automethod:: train

    .. automethod:: accuracy

    .. automethod:: confusion_matrix

Example: Heart Disease Prediction
---------------------------------

.. literalinclude:: ../tests/test_heart_bayes.py
   :pyobject: test_heart_bayes
   :lines: 2-
   :end-before: # Assert
   :dedent: 1