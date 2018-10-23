import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import random
import math
from timeit import Timer

TARGET_MSE = 0.02
MAX_EPOCHS = 100
LR = 0.02

# GradientDescentOptimizer
# MomentumOptimizer
# RMSPropOptimizer

def f(x, y):  # x, y ∈ [-1 1]
    return math.cos(x + 6*0.35*y) + 2*0.35*x*y


def genCoords(n):
    coords = []
    step = 1 / ((n-1) / 2.0)
    x = -1
    for _ in range(n):
        coords.append(x)
        x += step
    return coords


def genGridData(n, f):
    # Function output is scaled to fall between 0 and 1
    coords = genCoords(n)
    dataXY = []
    for x in coords:
        for y in coords:
            dataXY.append([x, y])
    dataZ = [(f(x, y)+1) / 2 for x, y in dataXY]
    dataZ = np.array(dataZ).reshape(n*n, 1)
    return np.concatenate((dataXY, dataZ), axis=1)


def genRandData(n, f):
    # Function output is scaled to fall between 0 and 1
    xy = [(random.random()*2 - 1, random.random()*2 - 1) for _ in range(n)]
    z = [(f(x, y)+1) / 2 for x, y in xy]
    return np.concatenate((np.array(xy), np.array(z).reshape(n, 1)), axis=1)


def pltData(data, color=''):
    g = int(math.sqrt(len(data)))
    plt.contour(
        data[:, 0].reshape((g, g)),
        data[:, 1].reshape((g, g)),
        data[:, 2].reshape((g, g)),
        10,
        colors=color
    )


def trainTest(n_hidden, epochs, lr, optimizer):
    # Generate training and test data
    # 10x10 and 9x9 grids: x, y ∈ [-1 1]
    trainData = genGridData(10, f)
    testData = genGridData(9, f)

    # 2 input neurons (x, y), 1 output (z)
    n_input = 2
    n_output = 1

    # X: input, Y: output
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # Weights and biases
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([n_hidden]))
    b2 = tf.Variable(tf.zeros([n_output]))

    # Hidden layer with sigmoid activation function
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

    # Output with linear activation function
    hy = tf.matmul(L2, W2) + b2


    cost = tf.reduce_mean((hy - Y)**2)
    optimizers = [
        tf.train.GradientDescentOptimizer(lr).minimize(cost),
        tf.train.MomentumOptimizer(lr, 0.9).minimize(cost),
        tf.train.RMSPropOptimizer(lr, momentum=0.9).minimize(cost)
    ]
    optimizer = optimizers[optimizer]
    init = tf.global_variables_initializer()

    # Randomize the order of training data
    np.random.seed(0)
    np.random.shuffle(trainData)
    x = trainData[:,:2]
    y = trainData[:,2].reshape(len(trainData), 1)

    # Keep track of costs so as to graph reduction in cost over epochs
    costs = []

    # At what epoch is the goal reached?
    goal_reached = -1

    with tf.Session() as sess:

        sess.run(init)
        for i in range(epochs):
            _, c = sess.run([optimizer, cost], feed_dict = {X: x, Y: y})
            costs.append(c)
            if c <= TARGET_MSE and goal_reached == -1:
                goal_reached = i
                # Stop and test once goal is reached
                break
        
        # Get prediction for testData, for comparison to actual f(x, y)
        pred = sess.run([hy], feed_dict = {X: testData[:,:2]})

    pred = [x[0] for x in pred[0]]
    mse = np.square(pred - testData[:,2]).mean()

    # Reshape prediction into format of testData for comparison [[x, y, z]]
    pred = np.concatenate((testData[:,:2], np.array(pred).reshape(len(pred), 1)),axis=1)
    return pred, mse, costs, goal_reached


# Part (b) 2.
def runExperiment(num_neurons, epochs, lr):
    results = []

    # Run experiments and get results for each optimizer
    for i in range(3):
        pred, mse, c, g = trainTest(num_neurons, epochs, lr, i)
        results.append((pred, mse, c, g))

    print("MSE with LR =", lr, "and", epochs, "epochs:")
    for i in range(3):
        print(results[i][1], ": Goal reached after", results[i][3], "epochs.")


runExperiment(8, 50000, LR)
# timeCPU(8, 1000, LR)