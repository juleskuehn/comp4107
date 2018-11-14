# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn


import numpy as np
import tensorflow as tf
import random
import imageio
from q2_kmeans import k_means
import datetime
import os
import scipy.linalg as sp

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0



def plot_clusters(points, centers, assignments, label=''):
    """
    Saves images to ./q2_plots/
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1], frameon=False)
    plt.text(0.05, 0.05, label, horizontalalignment='left', fontsize=12, 
        verticalalignment='bottom', transform=ax.transAxes)
    plt.scatter(points[:, 0], points[:, 1], c=assignments, s=50, alpha=0.5)
    plt.plot(centers[:, 0], centers[:, 1], 'ko', markersize=10)
    plt.plot(centers[:, 0], centers[:, 1], 'wx', markersize=10)
    plt.show()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(6, 4)
    fig.savefig('./q2_plots/' + label + '.png', dpi=100)
    plt.close('all')


# Testing
k = 28*28
n_features = 2 # Keep it 2 dimensional for visualization

points = x_train.reshape(60000, k)

history = k_means(points, k, verbose=True, sample_centers=True)

# Reduce points to 2D for graphing using SVD
# U, s, V = sp.svd(A, full_matrices=True)

# # Create animated gif of history
# for i, state in enumerate(history):
#     centers, assignments = state
#     # Reduce centers to 2D
#     plot_clusters(points, centers, assignments, label=str(i))


# time = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
# gif_path = './q2_plots/animation' + time + '.gif'

# # Slow down animation towards end
# durations = [1.5] + [1 * i / (len(history) + 1) + .2 for i in range(len(history) - 2)] + [3]

# with imageio.get_writer(gif_path, mode='I', duration=durations) as writer:
#     print("Generating animation...")
#     for i in range(len(history) + 1):
#         image = imageio.imread('./q2_plots/' + str(i) + '.png')
#         writer.append_data(image)

# for i in range(len(history) + 1):
#     os.remove('./q2_plots/' + str(i) + '.png')