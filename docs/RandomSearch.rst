Random Search for Hyperparameters
=================================

Class Reference
---------------

.. autoclass:: pykitml.RandomSearch

    .. automethod:: search

    .. automethod:: set_cost

    .. autoattribute:: best

Example: Tuning Feed-forward network for fashion-MNIST
-------------------------------------------------------

.. literalinclude:: ../tests/test_search.py
   :pyobject: test_search
   :lines: 3-
   :end-before: # Assert
   :dedent: 4