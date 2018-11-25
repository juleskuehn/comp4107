# COMP4107 Assignment 3
# Question 1 implementation by Yunkai Wang, student number 100968473
# Using the scikit-learn utilities to load the MNIST data, implement a Hopfield network that can classify the image data for a subset of the handwritten digits. Subsample the data to only include images of '1' and '5'. Here, correct classification means that if we present an image of a '1' an image of a '1' will be recovered; however, it may not be the original image owing to the degenerate property of this type of network. You are expected to document classification accuracy as a function of the number of images used to train the network. Remember, a Hopfield network can only store approximately 0.15N patterns with the "one shot" learning described in the lecture (slides 58-74).

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
seed = 11
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
    # pick random data from both 1's and 5's with equal amount to make
    # sure that the network will be able to learn both digits
    picked_ones = pick_random_data(ones, num_data)
    picked_fives = pick_random_data(fives, num_data)
    onesAndFives = picked_ones + picked_fives
    shuffle(onesAndFives)
    return onesAndFives


def getRepresentativeOnesAndFives(num_data=10, num_centers=2):
    allData = ones + fives
    shuffle(allData)
    # Prevent slow k_means by limiting number of samples
    k_means_samples = min(len(allData), 1000)
    smallData = allData[:k_means_samples]

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


# calculate the weight based on the given input vectors using Hebbian's rule or
# Storkey's rule based on the choice
def cal_weight(data, use_storkey_rule=False):
    p = len(data)
    W = np.zeros((input_len, input_len))

    for pixels, _ in data:
        # code for the bonus mark, Storkey's rule of learning
        if use_storkey_rule:
            # these variables relate to the terms used in Storkey's learning rule
            local_field = W.dot(pixels.transpose())
            term1 = np.outer(local_field, pixels) / input_len
            term2 = np.outer(pixels, local_field) / input_len
            W -= np.add(term1, term2)

        W += (pixels.transpose()).dot(pixels)
    W -= np.dot(np.identity(input_len), p)
    return W


# feed the input vector to the network with the weight and threshold value
def test(weight, input):
    changed = True # a variable that indicates if there exist any node which changes its state
    
    vector = input[0]
    indices = list(range(0, len(vector))) # do it for every node in the network

    while changed:  # repeat until converge
        changed = False

        # array to store new state after this iteration
        new_vector = np.array(
            [0 for _ in range(len(vector))]
        )
        shuffle(indices) # use different order every time
        
        for index in indices:
            s = compute_sum(weight, vector, index)
            new_vector[index] = 1 if s >= 0 else -1 # new state for the node
        
        changed = not np.allclose(vector, new_vector)
        vector = new_vector

    return vector


# compute the sum by adding up the weights of all active edges that connects to the
# given node
def compute_sum(weight, vector, node_index):
    return sum([weight[node_index][index] for index in range(len(vector)) if vector[index] == 1])


# Among the training data, find the data that's closest to the stable state and
# pick the label that corresponds to that data as the label for the state
def classify(vector, data):
    closestDis = float('inf')
    closestLabel = None

    for pixels, label in data:
        dis = np.linalg.norm(vector - pixels)
        if dis < closestDis:
            closestDis = dis
            closestLabel = label

    # print("Output vector, classified as", str(closestLabel))
    # printVector(vector)

    return closestLabel


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


def test_network(num_training_data=5, num_testing_data=10, use_storkey_rule=False, reprs=False):
    if reprs:
        # pick representative training data (closest to kmeans centers)
        trainingData = getRepresentativeOnesAndFives(num_training_data)
    else:
        trainingData = getRandomOnesAndFives(num_training_data)
    W = cal_weight(trainingData, use_storkey_rule)
    testData = getRandomOnesAndFives(num_testing_data)
    correct = 0 # number of correct identified image

    for pixels, actual_label in testData:
        # print("Input digit, with actual label", str(actual_label))
        # printVector(pixels)

        vector = test(W, pixels)
        label = classify(vector, trainingData)
        if actual_label == label: # correctly identified one image
            correct += 1

    # (2 * num_testing_data) because num_testing_data is for each of 1 and 5
    return correct / (2 * num_testing_data) # calculate the accuracy


# It seems like feeding the network with 5 of each digit will cause the network
# to forget everything, even if the original training data is tested. If I gave
# only 1 image of each digit, the network will do a relatively good job.
storkey = True
representative = False
numTest = 20
for i in range(1, 6):
    accuracy = test_network(i, numTest, use_storkey_rule=storkey, reprs=representative)
    print("number of training data for each digit:", i)
    print("number of test data for each digit:", numTest)
    print("Storkey:", storkey, "; Representative training data:", representative)
    print("accuracy:", accuracy)
    print("---")


# ----------------------------------------------------------------
# code for testing the network on the small example given in class
# input_len = 4 # testing small images

# # data found on slide 59 of Hopfield network
# x1 = np.array([1, -1, -1, 1])
# x2 = np.array([1, 1, -1, 1])
# x3 = np.array([-1, 1, 1, -1])
# # testing on the small example to find the problem
# trainingData = [[x1, 1], [x2, 2], [x3, 3]]
# W = cal_weight(trainingData, threshold=0)
# print(W)
# for pixels, actual_label in trainingData:
#     print("Start to test on pixels: ", pixels)
#     vector = test(W, pixels, threshold=0)
#     label = classify(vector, trainingData)
#     print(actual_label, label)