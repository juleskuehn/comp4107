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



# Testing
k = 10

history = k_means(points, k, verbose=True)

# # Create animated gif of history
# for i, state in enumerate(history):
#     centers, assignments = state
#     plot_clusters(points, centers, assignments, label=str(i))

# with imageio.get_writer('./q2_plots/animation'+datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")+'.gif', mode='I', duration=0.5) as writer:
#     print("Generating animation...")
#     for i in range(len(history)):
#         image = imageio.imread('./q2_plots/' + str(i) + '.png')
#         writer.append_data(image)
#         os.remove('./q2_plots/' + str(i) + '.png')