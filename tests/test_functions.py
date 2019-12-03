import numpy as np

from pykitml import _functions

eg_ws = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])

# =============================
# = Test activation functions =
# =============================

def test_sigmoid():
    expected_output = np.array([[0.52497919, 0.450166  , 0.57444252],
       [0.40131234, 0.62245933, 0.35434369]])

    assert np.allclose(_functions.sigmoid(eg_ws), expected_output)

def test_tanh():
    expected_output = np.array([[ 0.09966799, -0.19737532,  0.29131261],
       [-0.37994896,  0.46211716, -0.53704957]])

    assert np.allclose(_functions.tanh(eg_ws), expected_output)

def test_leakyrelu():
    expected_output = np.array([[ 0.1  , -0.002,  0.3  ],
       [-0.004,  0.5  , -0.006]])

    assert np.allclose(_functions.leakyrelu(eg_ws), expected_output)

def test_relu():
   expected_output = np.array([[0.1, 0, 0.3], [0, 0.5, 0]])

   assert np.allclose(_functions.relu(eg_ws), expected_output)

def test_softmax():
    expected_output = np.array([[0.33758454, 0.25008878, 0.41232669],
       [0.23373585, 0.57489742, 0.19136673]])

    assert np.allclose(_functions.softmax(eg_ws), expected_output)

# ===========================================
# = Test derivative of activation functions =
# ===========================================

def test_sigmoid_prime():
    activ = _functions.sigmoid(eg_ws)

    expected_output = np.array([[0.24937604, 0.24751657, 0.24445831],
       [0.24026075, 0.23500371, 0.22878424]])

    assert np.allclose(_functions.sigmoid_prime(eg_ws, activ), expected_output)

def test_tanh_prime():
    activ = _functions.tanh(eg_ws)

    expected_output = np.array([[0.99006629, 0.96104298, 0.91513696],
       [0.85563879, 0.78644773, 0.71157776]])

    assert np.allclose(_functions.tanh_prime(eg_ws, activ), expected_output)

def test_leakyrelu_prime():
    activ = _functions.leakyrelu(eg_ws)

    expected_output = np.array([[1.  , 0.01, 1.  ],
       [0.01, 1.  , 0.01]])

    assert np.allclose(_functions.leakyrelu_prime(eg_ws, activ), expected_output)

def test_relu_prime():
    activ = _functions.relu(eg_ws)

    expected_output = np.array([[1, 0, 1], [0, 1, 0]])

    assert np.allclose(_functions.relu_prime(eg_ws, activ), expected_output)

def test_softmax_prime():
    activ = _functions.leakyrelu(eg_ws)

    expected_output = np.array([[ 0.09    , -0.002004,  0.21    ],
       [-0.004016,  0.25    , -0.006036]])

    assert np.allclose(_functions.softmax_prime(eg_ws, activ), expected_output)

# =======================
# = Test cost functions =
# =======================

def test_mse():
   eg_output = np.array([0.1, 0.4, -0.1, 0.3])
   eg_target = np.array([0.2, 0.3, -0.5, 0.2])
   expected_output = np.array([0.005, 0.005, 0.08 , 0.005])

   assert np.allclose(_functions.mse(eg_output, eg_target), expected_output)

def test_cross_entropy():
   eg_output = np.array([0.3, 0.1, 0.9, 0.7])
   eg_target = np.array([1, 0, 1, 1])
   expected_output = np.array([1.2039728 , 0.10536052, 0.10536052, 0.35667494])

   assert np.allclose(_functions.cross_entropy(eg_output, eg_target), expected_output)

# =====================================
# = Test derivative of cost functions =
# =====================================

def test_mse_prime():
   eg_output = np.array([0.1, 0.4, -0.1, 0.3])
   eg_target = np.array([0.2, 0.3, -0.5, 0.2])
   expected_output = np.array([-0.1, 0.1, 0.4, 0.1])

   assert np.allclose(_functions.mse_prime(eg_output, eg_target), expected_output)

def test_cross_entropy_prime():
   eg_output = np.array([0.3, 0.1, 0.9, 0.7])
   eg_target = np.array([1, 0, 1, 1])
   expected_output = np.array([-3.33333333,  1.11111111, -1.11111111, -1.42857143])

   assert np.allclose(_functions.cross_entropy_prime(eg_output, eg_target), expected_output)

