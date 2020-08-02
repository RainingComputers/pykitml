from ._functions import pdist
import numpy as np

def smote(minority_data_points, k=1):
    '''
    SMOTE (Synthetic Minority Oversampling TEchnique).
    Used to generate more data points for minority class or imbalanced learning.

    Parameters
    ----------
    minority_data_points : numpy.array
        Inputs or data points corresponding to the minority class.
    k : int
        Number of neighbors to consider.

    Returns
    -------
    new_points : numpy.array
        New generated data points (Excluding data points passed to the 
        function). :code:`k*minority_data_points.shape[0]` points will be
        generated.
    '''
    npoints = minority_data_points.shape[0]
    nfeatures = minority_data_points.shape[1]
    
    # Calculate distance between each point and evry other point
    distances = pdist(minority_data_points, minority_data_points)
    
    # Get indices of closest k neigbours for each point
    indices = np.argsort(distances, axis=1)[:, 1:k+1]

    # Get the closest k neighbours for each point
    neighbours = minority_data_points[indices].squeeze()
    neighbours = neighbours.reshape(k*npoints, nfeatures)

    # Calculate diffrence between points and k neighbours
    minority_data_points_dups = minority_data_points[np.tile(np.arange(npoints).reshape(npoints, 1), k)]
    minority_data_points_dups = minority_data_points_dups.reshape(k*npoints, nfeatures)
    diff = neighbours - minority_data_points_dups

    # Create new data points
    random_floats = np.random.uniform(0, 1, (npoints*k))
    new_points = minority_data_points_dups + (diff.T*random_floats).T

    return new_points
