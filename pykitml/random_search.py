import random
import math


class RandomSearch:
    '''
    This class is used to search for hyperparameters. 
    '''

    def __init__(self):
        self._curr_cost = None
        self._best = False

    @property
    def best(self):
        '''
        If the last generated hyperparameters is the best so far.

        Note
        ----
        This property has to be used AFTER calling :py:func:`set_cost`
        '''
        return self._best

    def set_cost(self, cost):
        '''
        Set the cost for current hyperparameter.

        Parameters
        ----------
        cost : float
            The cost corresponding to current set of hyperparameters.
        '''
        self._curr_cost = cost

    def search(self, nsamples, nzoom, zoomratio, *args):
        '''
        Generator function to loop through randomly generated hyperparameters.
        Total number of hyperparameters sampled will be :code:`nsamples*nzoom`.
        First :code:`nsamples` points will be sampled, then the function will 
        'zoom in' around the best sample, and :code:`nsamples` more points will 
        be sampled. This will be repeated :code:`nzoom` times.
        The range for each hyperparameter should be passed as a list to
        :code:`*args`. The range should be :code:`[from, to, 'type']`, 
        for e.g. :code:`[0.8, 1, 'float']`. Three range types are available, 
        :code:`'float'`, :code:`'int'`, :code:`'log'`.

        Parameters
        ----------
        nsamples : int
            Number of hyperparameters to sample.
        nzoom : int
            Number of times to zoom in.
        zoomratio : float
            How much to zoom in.
        *args
            Range type for each hyperparameter.
        '''
        best_params = None
        min_cost = float('inf')
        range_types = args

        for z in range(nzoom):
            for i in range(nsamples):
                params = []
                # Generate hyperparameters
                for rtype in range_types:
                    l = rtype[0]
                    u = rtype[1]
                    if(rtype[2] == 'int'): params.append(random.randint(int(l), int(u)))
                    elif(rtype[2] == 'float'): params.append(random.uniform(l, u))
                    elif(rtype[2] == 'log'): params.append(10**random.uniform(l, u))

                print('Testing {}/{}, zoomlvl {},'.format(i+1, nsamples, z+1), 'params =', params)

                # Yield
                yield params

                # Track best ones
                if(self._curr_cost < min_cost):
                    min_cost = self._curr_cost
                    best_params = params
                    self._best = True
                else:
                    self._best = False

            # Zoom in around the best set of hyperparams
            new_range_types = []
            for best_param, rtype in zip(best_params, range_types):
                l = rtype[0]
                u = rtype[1]
                diff = u-l
                if(rtype[2] == 'log'): best_param = math.log10(best_param)
                new_l = best_param-(diff/zoomratio)
                new_u = best_param+(diff/zoomratio)
                new_range_types.append([new_l, new_u, rtype[2]])
            range_types = new_range_types


        # Print the best one
        print('\nSearch Finished')
        print('===============')
        print('Best params:', best_params)
        print('Best cost:', min_cost)