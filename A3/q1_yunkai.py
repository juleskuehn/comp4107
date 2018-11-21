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

    picked_ones = pick_random_data(ones, num_data)
    picked_fives = pick_random_data(fives, num_data)

    trainingData = picked_ones + picked_fives
    shuffle(trainingData)

    return trainingData

def getTestingData(num_data=10):
    # get the datas for 1 and 5
    ones = partition(x_test, y_test, 1)
    fives = partition(x_test, y_test, 5)

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

def compute_sum(weight, vector, node_index):
    sum = 0
    for index in range(len(vector)):
        if vector[index] == 1: # only sum up the nodes that are active
            sum += weight[node_index][index]
    return sum

num_training_data = 10 # number of training data feed to the network

# pick random number of vectors as input to the network
trainingData = getTrainingData(num_training_data)
W = cal_weight(trainingData)

# amoung the training data, find the data that's closest to the stable state and
# pick the label that corresponds to that data as the label for the state
def classify(vector, data):
    closestDis = float('inf')
    closestLabel = None

    for pixels, label in data:
        dis = np.linalg.norm(vector - pixels)
        if dis < closestDis:
            closestDis = dis
            closestLabel = label
    return closestLabel

# print(testData[0][0].reshape(28, 28))
# print(vector.reshape(28, 28))


# print(classify(vector, trainingData), trainingData[0][1])

testData = getTestingData()
for pixels, actual_label in testData:
    vector = test(W, pixels)
    label = classify(vector, trainingData)
    print(actual_label, label)
            