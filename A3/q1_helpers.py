# COMP 4107
# Fall 2018
# Assignment 3
# Yunkai Wang, student number 100968473
# Jules Kuehn, student number 100661464

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
from random import shuffle, sample
import random

from q2_kmeans import k_means


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# No need to separate training and test data here
x_data = np.concatenate((x_train, x_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)

input_len = 784 # 784 pixels for each image

# Set random seed for better reproducibility
seed = None
if seed:
    random.seed(seed)
    np.random.seed(seed)


# return part of the data that matches the given label
def partition(x_data, y_data, label):
    return [(np.array(x).ravel(), y) for (x, y) in zip(x_data, y_data) if y == label]


def thresholdData(data, threshold=127):
    return [
        (np.array([[1 if pixel <= threshold else -1 for pixel in pixels]]),
        label) for pixels, label in data
        ]


# get the thresholded data for 1 and 5
ones = thresholdData(partition(x_data, y_data, 1))
fives = thresholdData(partition(x_data, y_data, 5))


def pick_random_data(data, num_data):
    return list(np.array(data)[sample(list(range(len(data))), num_data)])


def getRandomOnesAndFives(num_data=10):
    print("Getting random ones and fives")
    # pick random data from both 1's and 5's with equal amount to make
    # sure that the network will be able to learn both digits
    picked_ones = pick_random_data(ones, num_data)
    picked_fives = pick_random_data(fives, num_data)
    onesAndFives = picked_ones + picked_fives
    shuffle(onesAndFives)
    return onesAndFives


# This version finds num_data closest points to each of the 2 centers
def getRepresentativeOnesAndFives(num_data=10, num_centers=2):
    print("Getting most similar (representative) training ones and fives")
    allData = ones + fives
    # Prevent slow k_means by limiting number of samples
    num_samples = min(len(allData), 1000)

    smallData = ones[:num_samples//2] + fives[:num_samples//2]
    shuffle(smallData)

    smallDataImages = [data[0].flatten() for data in smallData]

    print("Running k-means")
    # Initializing centers randomly within the range of each dimension
    # gives very poor results; instead sampling random training points
    centers, _ = k_means(smallDataImages, num_centers,
                            max_epochs=100, sample_centers=True, seed=seed)

    representatives = []
    # Find num_data closest points to each centers from smallData
    for center in centers:
        representatives += sorted(allData,
           key=lambda point: np.linalg.norm(point[0].flatten() - center))[:num_data]
    shuffle(representatives)
    return representatives
    

# This version finds the closest point to each of the num_data * 2 centers
def getCenterOnesAndFives(num_data=10):
    print("Getting training ones and fives that correspond to k-means centers")
    allData = ones + fives
    # Prevent slow k_means by limiting number of samples
    num_samples = min(len(allData), 1000)

    smallData = ones[:num_samples//2] + fives[:num_samples//2]
    shuffle(smallData)

    smallDataImages = [data[0].flatten() for data in smallData]

    print("Running k-means")
    # Initializing centers randomly within the range of each dimension
    # gives very poor results; instead sampling random training points
    centers, _ = k_means(smallDataImages, num_data * 2,
                            max_epochs=100, sample_centers=True, seed=seed)

    representatives = []
    # Find num_data closest points to each centers from smallData
    for center in centers:
        representatives.append(min(allData,
           key=lambda point: np.linalg.norm(point[0].flatten() - center)))
    shuffle(representatives)
    return representatives


def printVector(vector):
    grid = list(np.array(vector).reshape(28, 28))
    for row in grid:
        row_str = ''
        for char in row:
            if char == 1:
                row_str += ' '
            else:
                row_str += '#'
        print(row_str)
    print('-' * 28)
