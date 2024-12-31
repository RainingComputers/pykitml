Deep Q Learning
===============

DQNAgent Class
--------------

.. autoclass:: pykitml.DQNAgent
    
    .. automethod:: __init__

    .. automethod:: train

    .. automethod:: exploit

    .. automethod:: plot_performance

.. _environment:

Environment Class
-----------------

.. autoclass:: pykitml.Environment
    
    .. automethod:: reset

    .. automethod:: step

    .. automethod:: close

    .. automethod:: render

Example : Cartpole using gymnasium
----------------------------------

.. literalinclude:: ../tests/test_cartpole_dqn.py
   :pyobject: test_cartpole
   :lines: 3-
   :dedent: 4
