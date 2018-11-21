# COMP4107 Assignment 3
# quesion 1 implementation by Yunkai Wang, student number 100968473
# Using the scikit-learn utilities to load the MNIST data, implement a Hopfield network that can classify the image data for a subset of the handwritten digits. Subsample the data to only include images of '1' and '5'. Here, correct classification means that if we present an image of a '1' an image of a '1' will be recovered; however, it may not be the original image owing to the degenerate property of this type of network. You are expected to document classification accuracy as a function of the number of images used to train the network. Remember, a Hopfield network can only store approximately 0.15N patterns with the "one shot" learning described in the lecture (slides 58-74).

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
from random import shuffle, sample

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_len = 784 # 784 pixels for each image

# return part of the data that matches the given label
def partition(x_data, y_data,label):
    return [(np.array(input).ravel(), label) for (input, output) in zip(x_data, y_data) if output == label]

def pick_random_data(data, num_data):
    random_data = []
    indices = sample(list(range(len(data))), num_data)
    for index in indices:
        random_data.append(data[index])
    return random_data

def getTrainingData(num_data=10):
    # get the datas for 1 and 5
    ones = partition(x_train, y_train, 1)
    fives = partition(x_train, y_train, 5)

    # pick random data from both 1's and 5's with equal amount to make
    # sure that the network will be able to learn both digits
    picked_ones = pick_random_data(ones, num_data)
    picked_fives = pick_random_data(fives, num_data)

    trainingData = picked_ones + picked_fives
    shuffle(trainingData)

    return trainingData

def getTestingData(num_data=5):
    # get the datas for 1 and 5
    ones = partition(x_test, y_test, 1)
    fives = partition(x_test, y_test, 5)

    # pick random data for both 1's and 5's to check the network's accuracy
    picked_ones = pick_random_data(ones, num_data)
    picked_fives = pick_random_data(fives, num_data)
    testingData = picked_ones + picked_fives
    shuffle(testingData)

    return testingData

# calculate the weight based on the given input vectors using Hebbian's rule
def cal_weight(data, threshold = 127):
    p = len(data)
    W = np.zeros((input_len, input_len))

    for pixels, _ in data:
        # threshold the given vector
        x = np.array(
            [[1 if pixel >= threshold else -1 for pixel in pixels]]
        )
        W += (x.transpose()).dot(x)
    W -= np.dot(np.identity(input_len), p)
    return W

# feed the input vector to the network with the weight and threshold value
def test(weight, input, threshold=127):
    changed = True # a variable that indicates if there exist any node which changes its state
    
    # threshold the input vector
    vector = np.array(
        [1 if pixel >= threshold else -1 for pixel in input]
    )
    # print(vector.reshape(28, 28))
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
        # print(vector.reshape(28, 28))
    return vector

# compute the sum by adding up the weights of all active edges that connects to the
# given node
def compute_sum(weight, vector, node_index):
    sum = 0
    for index in range(len(vector)):
        if vector[index] == 1: # only sum up the nodes that are active
            sum += weight[node_index][index]
    return sum

# amoung the training data, find the data that's closest to the stable state and
# pick the label that corresponds to that data as the label for the state
# PROBLEM: seems like it always classsify all the digits as the same digit no
# matter what
def classify(vector, data):
    closestDis = float('inf')
    closestLabel = None

    for pixels, label in data:
        dis = np.linalg.norm(vector - pixels)
        if dis < closestDis:
            closestDis = dis
            closestLabel = label
    return closestLabel

num_training_data = 10 # number of training data feed to the network

# pick random number of vectors as input to the network
trainingData = getTrainingData(num_training_data)
W = cal_weight(trainingData)

testData = getTestingData()
for pixels, actual_label in trainingData:
    vector = test(W, pixels)
    label = classify(vector, trainingData)
    print(actual_label, label)
            

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