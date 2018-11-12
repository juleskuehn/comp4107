# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn


import numpy as np
import tensorflow as tf
import random
import imageio
from q2_kmeans import k_means


def rand_points(num_points, num_dim):
    """
    For testing purposes.
    Returns a 2d numpy array where each row is a point.
    """
    return np.array([
        [random.uniform(0, 100) for _ in range(num_dim)]
        for _ in range(num_points)
    ])


def plot_clusters(points, centers, assignments, label=''):
    """
    Saves images to ./q2_plots/
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.text(0.9, 0.1, label, horizontalalignment='right', fontsize=28, 
        verticalalignment='bottom', transform=ax.transAxes)
    plt.scatter(points[:, 0], points[:, 1], c=assignments, s=60, alpha=0.7)
    plt.plot(centers[:, 0], centers[:, 1], 'kx', markersize=15)
    plt.show()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6, 4)
    fig.savefig('./q2_plots/' + label + '.png', dpi=100)
    plt.close('all')


# Testing
k = 5
points = rand_points(100, 2)
history = k_means(points, k, verbose=True)

# Create animated gif of history
for i, state in enumerate(history):
    centers, assignments = state
    plot_clusters(points, centers, assignments, label=str(i))

with imageio.get_writer('./q2_plots/animation.gif', mode='I', duration=0.5) as writer:
    for i in range(len(history)):
        image = imageio.imread('./q2_plots/' + str(i) + '.png')
        writer.append_data(image)