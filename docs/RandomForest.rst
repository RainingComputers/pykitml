Random Forest
=============

Class Reference
---------------

.. autoclass:: pykitml.RandomForest
    
    .. automethod:: __init__

    .. automethod:: feed

    .. automethod:: get_output

    .. automethod:: get_output_onehot

    .. automethod:: train

    .. automethod:: accuracy

    .. automethod:: confusion_matrix

    .. autoattribute:: trees

Example: Predicting Income
--------------------------

.. literalinclude:: ../tests/test_adult_forest.py
   :pyobject: test_adult_forest
   :lines: 3-
   :end-before: # Assert
   :dedent: 1