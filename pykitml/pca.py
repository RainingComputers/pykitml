import numpy as np

class PCA:
    '''
    This class implements Principle Component Analysis.
    '''
    def __init__(self, data_points, no_components):
        '''
        This class implements Principle Component Analysis, used for 
        dimensionality reduction.

        Parameters
        ----------
        data_points : numpy.array
            The dataset to perform PCA i.e. dimensionality reduction on.
        no_components : int
            Number of principle components to use.
        '''
        # Calculate covariance matrix
        covariance_matrix = (data_points.T) @ data_points;
        covariance_matrix = covariance_matrix/data_points.shape[0]

        # Perform Singular Value Decomposition on the comvariance matrix
        u, s, v = np.linalg.svd(covariance_matrix, full_matrices=True)

        # Calculate amount of variance retained
        self._retention = np.sum(s[0:no_components])/np.sum(s)

        # The transformation matrix for PCA
        self._transform = u[:, 0:no_components]

    def transform(self, data_points):
        '''
        Transforms the input dataset to lower dimensions.
        
        Parameters
        ----------
        data_points : numpy.array
            The input dataset.

        Returns
        -------
        transformed_data_points : numpy.array
            The transformed input.
        '''
        # Transform the datapoints using principle components
        return data_points@self._transform

    def inverse_transform(self, pca_points):
        '''
        Gets the original dataset from transformed points.

        Parameters
        ----------
        pca_points : numpy.array
            The trasformed points.

        '''
        # Transform from principle components back to approx feature
        return pca_points @ (self._transform.T)

    @property
    def retention(self):
        '''
        Returns the amount of variance retained, between 0 and 1.
        '''
        return round(self._retention, 2)