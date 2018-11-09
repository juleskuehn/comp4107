# https://learningtensorflow.com/lesson6/

import tensorflow as tf
import numpy as np


def create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed):
    np.random.seed(seed)
    slices = []
    centroids = []
    # Create samples for each cluster
    for i in range(n_clusters):
        samples = tf.random_normal((n_samples_per_cluster, n_features),
                                   mean=0.0, stddev=5.0, dtype=tf.float32, seed=seed, name="cluster_{}".format(i))
        current_centroid = (np.random.random((1, n_features))
                            * embiggen_factor) - (embiggen_factor/2)
        centroids.append(current_centroid)
        samples += current_centroid
        slices.append(samples)
    # Create a big "samples" dataset
    samples = tf.concat(slices, 0, name='samples')
    centroids = tf.concat(centroids, 0, name='centroids')
    return centroids, samples


def plot_clusters(all_samples, centroids, n_samples_per_cluster, label='fig', num_clusters=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plot out the different clusters
    # Choose a different colour for each cluster
    colour = ['red', 'green', 'blue', 'grey', 'yellow', 'purple', 'cyan', 'magenta']
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.text(0.9, 0.1, label, horizontalalignment='right', fontsize=44, 
        verticalalignment='bottom', transform=ax.transAxes)
    if num_clusters == '':
        num_clusters = len(centroids)
    for i in range(num_clusters):
        # Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = all_samples[i *
                              n_samples_per_cluster:(i+1)*n_samples_per_cluster]
        plt.scatter(samples[:, 0], samples[:, 1], c=colour[i])
    for i, centroid in enumerate(centroids):
        # Also plot centroid
        plt.plot(centroid[0], centroid[1], markersize=35,
                 marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30,
                 marker="x", color='w', mew=5)

    plt.show()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(label + '.png', dpi=100)
    plt.close('all')


def choose_random_centroids(samples, n_clusters, seed):
    # Step 0: Initialisation: Select `n_clusters` number of random points
    # from the samples
    n_samples = tf.shape(samples)[0] # Number of dimensions
    random_indices = tf.random_shuffle(tf.range(0, n_samples), seed=seed) # Random sampling
    begin = [0, ]
    size = [n_clusters, ] 
    size[0] = n_clusters
    # Get the first n_clusters of random sample indices
    centroid_indices = tf.slice(random_indices, begin, size)
    # Get the samples themselves
    initial_centroids = tf.gather(samples, centroid_indices)
    return initial_centroids


def assign_to_nearest(samples, centroids):
    # Finds the nearest centroid for each sample

    # START from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    expanded_vectors = tf.expand_dims(samples, 0)
    expanded_centroids = tf.expand_dims(centroids, 1)
    distances = tf.reduce_sum( tf.square(
               tf.subtract(expanded_vectors, expanded_centroids)), 2)
    mins = tf.argmin(distances, 0)
    # END from http://esciencegroup.com/2016/01/05/an-encounter-with-googles-tensorflow/
    nearest_indices = mins
    return nearest_indices


def update_centroids(samples, nearest_indices, n_clusters):
    # Updates the centroid to be the mean of all samples associated with it.
    nearest_indices = tf.to_int32(nearest_indices)
    partitions = tf.dynamic_partition(samples, nearest_indices, n_clusters)
    new_centroids = tf.concat([tf.expand_dims(tf.reduce_mean(partition, 0), 0) for partition in partitions], 0)
    return new_centroids