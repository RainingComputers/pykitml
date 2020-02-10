import os
import cProfile
from unittest.mock import patch
from functools import wraps

import matplotlib.pyplot as plt
from graphviz import Digraph 


def _profile(test_func):
    '''
    Calls test function and profiles it.

    Parameters
    ----------
    test_func : function
        The function to test and profile.
    '''
    # Call the test function and profile it
    profiler = cProfile.Profile()
    profiler.runcall(test_func)
    profiler.dump_stats(test_func.__name__+'.dat') 


def pktest_graph(test_func):
    '''
    To test and profile function under pytest. Will prevent 
    :code:`matplotlib.pyplot.show()` from blocking other tests.

    Parameters
    ----------
    test_func : function
        The function to test and profile.
    '''
    # Create wrapper function for testing and profiling in pytest
    @wraps(test_func)
    def test_wrapper():
        with patch('matplotlib.pyplot.show') as show_func, patch('graphviz.Digraph.view') as view_func:
            _profile(test_func)
        
            # Test if graph worked
            if "PYTEST_CURRENT_TEST" in os.environ:
                assert show_func.called
                plt.close()
                plt.clf()

    return test_wrapper


def pktest_nograph(test_func):
    '''
    To test and profile function under pytest.

    Parameters
    ----------
    test_func : function
        The function to test and profile.
    '''
    # Create wrapper function for testing and profiling in pytest
    @wraps(test_func)
    def test_wrapper():
        _profile(test_func)

    return test_wrapper