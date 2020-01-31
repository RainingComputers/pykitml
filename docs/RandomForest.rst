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

Example: Banknote Authentication
--------------------------------

**Dataset**

:ref:`banknote_dataset`

**Training**

.. literalinclude:: ../tests/test_banknote_forest.py
   :pyobject: test_banknote_forest
   :lines: 2-
   :end-before: # Assert
   :dedent: 1

**Predict banknote validity with variance, skewness, curtosis, entropy: 
-2.3, -9.3, 9.37, -0.86**

.. literalinclude:: ../tests/test_banknote_forest.py
   :pyobject: test_predict
   :lines: 2-
   :dedent: 1