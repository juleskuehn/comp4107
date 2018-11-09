# https://learningtensorflow.com/lesson6/

import tensorflow as tf

from functions import *

n_features = 2
n_clusters = 5
k = 20
n_samples_per_cluster = 100
seed = 700
embiggen_factor = 70


data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, k, seed)
nearest_indices = assign_to_nearest(samples, initial_centroids)
updated_centroids = update_centroids(samples, nearest_indices, k)

epochs = 50

model = tf.global_variables_initializer()

# for animated gif
filenames = ['0.png']

with tf.Session() as session:
    sample_values = session.run(samples)
    centroid_values = session.run(initial_centroids)
    plot_clusters(sample_values, centroid_values, n_samples_per_cluster, label='0', num_clusters=n_clusters)
    for step in range(epochs):
        # sample_values = session.run(samples)
        nearest_indices = assign_to_nearest(samples, updated_centroids)
        updated_centroids = update_centroids(samples, nearest_indices, k)
    updated_centroid_value = session.run(updated_centroids)
    plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster, label=str(step+1), num_clusters=n_clusters)
    filenames.append(str(step+1) +'.png')
    print('epoch ' + str(step+1))
    print(updated_centroid_value)

# import imageio
# with imageio.get_writer('animation.gif', mode='I', duration=0.5) as writer:
#     for filename in filenames:
#         image = imageio.imread(filename)
#         writer.append_data(image)