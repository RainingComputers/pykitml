Normalization/Feature-scaling
=============================

Min-Max Normalization
---------------------

.. autofunction:: pykitml.get_minmax

.. autofunction:: pykitml.normalize_minmax

.. autofunction:: pykitml.denormalize_minmax

**Example**

>>> import numpy as np
>>> import pykitml as pk
>>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
>>> min_array, max_array = pk.get_minmax(a)
>>> normalized_a = pk.normalize_minmax(a, min_array, max_array)
>>> normalized_a
array([[0.        , 0.        , 0.        , 0.        ],
       [0.33333333, 0.33333333, 0.33333333, 0.33333333],
       [0.66666667, 0.66666667, 0.66666667, 0.66666667],
       [1.        , 1.        , 1.        , 1.        ]])
>>> pk.denormalize_minmax(normalized_a, min_array, max_array)
array([[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.],
       [ 9., 10., 11., 12.],
       [13., 14., 15., 16.]])

You can also only normalize/denormalize specific columns,

>>> import numpy as np
>>> import pykitml as pk
>>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
>>> min_array, max_array = pk.get_minmax(a)
>>> normalized_a = pk.normalize_minmax(a, min_array, max_array, cols=[0, 2])
>>> normalized_a
array([[ 0.        ,  2.        ,  0.        ,  4.        ],
       [ 0.33333333,  6.        ,  0.33333333,  8.        ],
       [ 0.66666667, 10.        ,  0.66666667, 12.        ],
       [ 1.        , 14.        ,  1.        , 16.        ]])
>>> pk.denormalize_minmax(normalized_a, min_array, max_array, cols=[0, 2])
array([[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.],
       [ 9., 10., 11., 12.],
       [13., 14., 15., 16.]])

Mean Normalization
------------------

.. autofunction:: pykitml.get_meanstd

.. autofunction:: pykitml.normalize_mean

.. autofunction:: pykitml.denormalize_mean

**Example**

>>> import numpy as np
>>> import pykitml as pk
>>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
>>> array_mean, array_stddev = pk.get_meanstd(a)
>>> normalized_a = pk.normalize_mean(a, array_mean, array_stddev)
>>> normalized_a
array([[-1.34164079, -1.34164079, -1.34164079, -1.34164079],
       [-0.4472136 , -0.4472136 , -0.4472136 , -0.4472136 ],
       [ 0.4472136 ,  0.4472136 ,  0.4472136 ,  0.4472136 ],
       [ 1.34164079,  1.34164079,  1.34164079,  1.34164079]])
>>> pk.denormalize_mean(normalized_a, array_mean, array_stddev)
array([[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.],
       [ 9., 10., 11., 12.],
       [13., 14., 15., 16.]])

You can also only normalize/denormalize specific columns,

>>> import numpy as np
>>> import pykitml as pk
>>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])
>>> array_mean, array_stddev = pk.get_meanstd(a)
>>> normalized_a = pk.normalize_mean(a, array_mean, array_stddev, cols=[0,2])
>>> normalized_a
array([[-1.34164079,  2.        , -1.34164079,  4.        ],
       [-0.4472136 ,  6.        , -0.4472136 ,  8.        ],
       [ 0.4472136 , 10.        ,  0.4472136 , 12.        ],
       [ 1.34164079, 14.        ,  1.34164079, 16.        ]])
>>> pk.denormalize_mean(normalized_a, array_mean, array_stddev, cols=[0,2])
array([[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.],
       [ 9., 10., 11., 12.],
       [13., 14., 15., 16.]])

