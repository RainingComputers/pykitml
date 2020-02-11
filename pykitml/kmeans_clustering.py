import numpy as np
import tqdm
import matplotlib.pyplot as plt

from . import _functions
from . import normalize

def kmeans(training_data, nclusters, max_iter=1000, trials=50):
    '''
    Identifies cluster centres on training data using k-means.

    Parameters
    ----------
    training_data : numpy.array
        Numpy array containing training data.
    nclusters : int
        Number of cluster to find.
    max_iter : int
        Maximum number of iterations to run per trial.
    trials : int
        Number of times k-means should run, each with different
        random initialization.

    Returns
    -------
    clusters : numpy.array
        Numpy array containing cluster centres.
    cost : numpy.array
        The cost of converged cluster centres.

    '''
    
    # Keep track of trial with the least cost
    min_cost = np.float('infinity')
    distances = None
    clusters_min_cost = None
    clusters = None

    # Keep log of maximum number of iterations for convergence
    max_iter_log = 0
    
    pbar = tqdm.trange(0, trials, ncols=80, unit='trials')
    for trial in pbar:
        # Use kmeans++ to initialize cluster centres
        clusters = np.zeros((nclusters, training_data.shape[1]))
        
        # First cluster centre is random
        index = np.random.randint(training_data.shape[0], size=1)
        clusters[0] = training_data[index]
        
        # Loop for rest of cluster centres
        for i in range(1, nclusters):
            # Calculate distance between every data point and previous cluster centre
            prev_cluster_dists = _functions.pdist(clusters[i-1], training_data).squeeze()
            # Normalize distances
            prev_cluster_dists = prev_cluster_dists/prev_cluster_dists.sum()
            
            # Sample index with probability distribution proportional to distances
            index = np.random.choice(training_data.shape[0], 1, p=prev_cluster_dists)
            
            # Assign next cluster centre
            clusters[i] = training_data[index]
        
        # Start kmeans, Keep looping and moving the cluster points to mean
        for iteration in range(max_iter):
            new_clusters = np.zeros((nclusters, training_data.shape[1]))

            # Calculate distances between clusters and every point in training data 
            distances = _functions.pdist(training_data, clusters)

            # Assign clusters index to each data point
            cluster_assignments = np.argmin(distances, axis=1)

            # Move cluster by taking mean of all the points assigned to that cluster
            for i in range(nclusters):
                cluster_points = training_data[cluster_assignments==i]
                if(cluster_points.shape[0] == 0): continue
                new_clusters[i] = np.mean(cluster_points, axis=0)
            
            # Check for convergence
            if(np.abs(new_clusters-clusters)==0).all(): break

            # Assign new clusters
            clusters = new_clusters

        # Select cluster centres with least cost
        cost = np.mean(np.min(distances, axis=1))
        if(cost < min_cost):
            clusters_min_cost = clusters
            min_cost = cost

        # Update maximum iterations for convergence
        if(iteration > max_iter_log):
            max_iter_log = iteration

        # Update progress bar
        pbar.set_postfix(cost=min_cost, max_it=max_iter_log)

    return clusters_min_cost, min_cost
