# ================================================
# = Test normalization/feature-scaling functions =
# ================================================


import numpy as np

import pykitml as pk

eg_array = np.array([
    [0.1,   0.3434, 1.3434, 3],
    [1.2,   4.54,   6.7,    3.456],
    [5.678, 2.345,  2.453,  8.345],
    [2.3,   6.2,    8.3,    1.2]
])


def test_minmax():
    expected_output = (np.array([0.1, 0.3434, 1.3434, 1.2]),
                       np.array([5.678, 6.2, 8.3, 8.345]))

    assert np.allclose(pk.get_minmax(eg_array), expected_output)


def test_normalize():
    array_min, array_max = pk.get_minmax(eg_array)

    norm_array = pk.normalize_minmax(eg_array, array_min, array_max)
    denorm_array = pk.denormalize_minmax(norm_array, array_min, array_max)

    assert np.allclose(denorm_array, eg_array)
