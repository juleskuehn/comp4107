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
        
        # Get prediction for testData, for comparison to actual f(x, y)
        pred = sess.run([hy], feed_dict = {X: testData[:,:2]})

    pred = [x[0] for x in pred[0]]
    mse = np.square(pred - testData[:,2]).mean()

    # Reshape prediction into format of testData for comparison [[x, y, z]]
    pred = np.concatenate((testData[:,:2], np.array(pred).reshape(len(pred), 1)),axis=1)
    return pred, mse, costs, goal_reached


def runExperiment(num_neurons, epochs, lr):
    results = []

    # Run experiments and get results for each optimizer
    for i in range(3):
        pred, mse, c, g = trainTest(num_neurons, epochs, lr, i)
        results.append((pred, mse, c, g))

    print("MSE with LR =", lr, "and", epochs, "epochs:")
    for i in range(3):
        print(results[i][1], ": Goal reached after", results[i][3], "epochs.")

    # Title offsets
    off = 0.7

    plt.figure(1)
    plt.subplot(311)
    plt.title('GradientDescentOptimizer', y=off)
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.plot(range(0, epochs), results[0][2], 'k')
    if results[0][3] != -1:
        plt.plot((results[0][3],results[0][3]),(0,100),'k--')
    plt.ylim(0,0.6)

    plt.subplot(312)
    plt.title('MomentumOptimizer', y=off)
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.plot(range(0, epochs), results[1][2], 'k')
    if results[1][3] != -1:
        plt.plot((results[1][3],results[1][3]),(0,100),'k--')
    plt.ylim(0,0.6)
    
    plt.subplot(313)
    plt.title('RMSPropOptimizer', y=off)
    plt.ylabel('MSE')
    plt.xlabel('Epochs')
    plt.plot(range(0, epochs), results[2][2], 'k')
    if results[2][3] != -1:
        plt.plot((results[2][3],results[2][3]),(0,100),'k--')
    plt.ylim(0,0.6)
    plt.show()


def timeCPU(num_neurons, epochs, lr):
    gradientTime = Timer(lambda: trainTest(num_neurons, epochs, lr, 0))
    momentumTime = Timer(lambda: trainTest(num_neurons, epochs, lr, 1))
    rmsPropTime = Timer(lambda: trainTest(num_neurons, epochs, lr, 2))

    gradientTime = gradientTime.timeit(number=5) / (epochs*5)
    momentumTime = momentumTime.timeit(number=5) / (epochs*5)
    rmsPropTime = rmsPropTime.timeit(number=5) / (epochs*5)

    names = ('GD', 'Momentum', 'RMSProp')
    
    plt.bar(range(3), [gradientTime, momentumTime, rmsPropTime])
    plt.xticks(range(3), names)
    plt.ylabel('CPU time per epoch (s)')
    
    plt.show()


# runExperiment(8, 200, LR)
timeCPU(8, 1000, LR)