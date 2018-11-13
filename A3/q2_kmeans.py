# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn


import numpy as np
import math
import random


def init_centers_random(points, k):
    """ 
    points: a 2d numpy array where each row is a point.
    k: number of cluster centers for k-means.
    Returns a 2d numpy array where each row is a cluster center,
    generated from random numbers within min/max for each dimension in points.
    """
    mins = [min(points[:, i]) for i in range(len(points[0, :]))]
    maxs = [max(points[:, i]) for i in range(len(points[0, :]))]
    return np.array([
        [random.uniform(mins[i], maxs[i]) for i in range(len(mins))]
        for _ in range(k)
    ])


def init_centers_sampling(points, k):
    """ 
    points: a 2d numpy array where each row is a point.
    k: number of cluster centers for k-means.
    Returns a 2d numpy array where each row is a cluster center,
    found by randomly sampling from points.
    """
    return random.sample(list(points), k)


def assign_nearest(points, centers):
    """
    points: a 2d numpy array where each row is a point.
    centers: a 2d numpy array where each row is a cluster center.
    Returns an array of indices of closest center for each point in points.
    """
    # for each point, find closest center
    return [
        np.argmin(
            [np.linalg.norm(point - center) for center in centers]
        ) for point in points
    ]


def update_centers(points, assignments, centers):
    """
    points: a 2d numpy array where each row is a point.
    assignments: array of indices of closest center for each point in points.
    centers: a 2d numpy array where each row is a cluster center.
    Returns none; updates centers.
    """
    points_by_center = [[] for _ in range(len(centers))]
    # Gather all points assigned to a given center
    for i, point in enumerate(points):
        points_by_center[assignments[i]].append(point)
    # Assign centers to mean of assigned points, if any assigned
    for i, gathered_points in enumerate(points_by_center):
        if len(gathered_points) > 0:
            centers[i] = np.average(gathered_points, axis=0)


def k_means(points, k, max_epochs=1000, verbose=False, sample_centers=False):
    """ 
    points: a 2d numpy array where each row is a point.
    k: number of cluster centers for k-means.
    Returns a 2d numpy array where each row is a cluster center.
    If verbose flag is True:
    Returns an array of tuples of (centers, assignments).
    """
    # Initialization
    if sample_centers:
        centers = init_centers_sampling(points, k)
    else:
        centers = init_centers_random(points, k)
    assignments = assign_nearest(points, centers)

    # For visualization and debugging, keep history of centers and assignments
    history = [(np.copy(centers), assignments)]

    # Training: Iterate until convergence (when assignments don't change)
    last_assignments = assignments
    for step in range(1, max_epochs + 1):
        update_centers(points, assignments, centers)
        assignments = assign_nearest(points, centers)
        # Convergence?
        if last_assignments == assignments:
            break
        last_assignments = assignments
        if verbose:
            history.append((np.copy(centers), assignments))

    if verbose:
        if step < max_epochs:
            print('Converged after', step, 'iterations.')
        else:
            print('Never converged - stopped after', max_epochs, 'iterations.')

    if verbose:
        return history
    else:
        return centers

