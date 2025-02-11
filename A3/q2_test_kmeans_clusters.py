# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn


import numpy as np
import tensorflow as tf
import random
import imageio
from q2_kmeans import k_means, calc_betas
import datetime
import os


def rand_points(num_points, num_dim):
    """
    For testing purposes.
    Returns a 2d numpy array where each row is a point.
    """
    return np.array([
        [random.uniform(0, 100) for _ in range(num_dim)]
        for _ in range(num_points)
    ])


def plot_clusters(points, centers, assignments, label='', betas=None):
    """
    Saves images to ./q2_plots/
    """
    if not betas:
        betas = [10] * len(centers)
    else:
        betas = [10/b for b in betas]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1], frameon=False)
    plt.text(0.05, 0.05, label, horizontalalignment='left', fontsize=12, 
        verticalalignment='bottom', transform=ax.transAxes)
    plt.scatter(points[:, 0], points[:, 1], c=assignments, s=50, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='k', s=betas, alpha=0.25)
    plt.scatter(centers[:, 0], centers[:, 1], c='w', s=10, alpha=1)
    plt.show()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6, 4)
    fig.savefig('./q2_plots/' + label + '.png', dpi=100)
    plt.close('all')


# Creates pseudo-random clustered data
def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed=None):
    if seed:
        np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32,
                                   seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features))
                            * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    with tf.Session() as session:
        return samples.eval(), centroids.eval()


# Testing
k = 5
n_features = 2 # Keep it 2 dimensional for visualization

n_samples_per_cluster = 100
# seed = 700
embiggen_factor = 75 # A higher value makes for denser clusters
points, true_centers = create_samples(
                        k, n_samples_per_cluster, n_features, embiggen_factor)

history = k_means(points, k, verbose=True, sample_centers=True)


# Create animated gif of history
for i, state in enumerate(history):
    centers, assignments, betas = state
    plot_clusters(points, centers, assignments, label=str(i), betas=betas)

# Finally, show actual cluster centers
centers = true_centers
# Show actual clusters (may change colors)
assignments = []
for i in range(k):
    assignments += [i for _ in range(n_samples_per_cluster)]

betas = calc_betas(points, assignments, centers)

plot_clusters(points, centers, assignments, label=str(len(history)), betas=betas)

time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
gif_path = './q2_plots/animation' + time + '.gif'

# Slow down animation towards end
durations = [1.5] + [1 * i / (len(history) + 1) + .2 for i in range(len(history) - 1)] + [3]

with imageio.get_writer(gif_path, mode='I', duration=durations) as writer:
    print("Generating animation...")
    for i in range(len(history) + 1):
        image = imageio.imread('./q2_plots/' + str(i) + '.png')
        writer.append_data(image)

for i in range(len(history) + 1):
    os.remove('./q2_plots/' + str(i) + '.png')