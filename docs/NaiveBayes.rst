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

**Dataset**

:ref:`heart_dataset`

**Training**

.. literalinclude:: ../tests/test_heart_bayes.py
   :pyobject: test_heart_bayes
   :lines: 3-
   :end-before: # Assert
   :dedent: 4

**Predict heartdisease for a person with 
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal:
67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3**

.. literalinclude:: ../tests/test_heart_bayes.py
   :pyobject: test_predict_heart_bayes
   :lines: 3-
   :dedent: 4

**Confusion Matrix**

.. image:: ./demo_pics/bayes_confusion_matrix.png