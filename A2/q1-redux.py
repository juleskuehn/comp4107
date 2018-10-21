import numpy as np
import scipy.linalg as sp
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
import random
import pandas as pd

TOLERANCE = 0.02  # MSE
MAX_EPOCHS = 100
LR = 0.02


def satisfied(fpx, fx, ε):
    return np.linalg.norm(fpx - fx, ord=1) < ε


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


def trainTest(n_hidden=50, epochs=3000, lr=0.5):
    trainData = genGridData(10, f)
    testData = genGridData(9, f)
    # validationData = genRandData(100, f)

    # Hyperparamters
    n_input = 2
    n_output = 1
    display_step = 100

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
    b1 = tf.Variable(tf.zeros([n_hidden]))
    b2 = tf.Variable(tf.zeros([n_output]))

    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
    hy = tf.matmul(L2, W2) + b2
    cost = tf.reduce_mean((hy - Y)**2)
    optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

    init = tf.global_variables_initializer()

    td = trainData[:,:]
    np.random.shuffle(td)
    
    with tf.Session() as sess:
        sess.run(init)
        x = td[:,:2]
        y = td[:,2].reshape(len(td), 1)
        for step in range(epochs):
            _, c = sess.run([optimizer, cost], feed_dict = {X: x, Y: y})
            # pred = sess.run([hy], feed_dict = {X: validationData[:,:2]})
            # if satisfied(pred, validationData[:,2], TOLERANCE):
            #     break
            if step % display_step == 0:
                print("Cost: ", c)
            
        pred = sess.run([hy], feed_dict = {X: testData[:,:2]})

    pred = [x[0] for x in pred[0]]
    mse = np.square(pred - testData[:,2]).mean()
    pred = np.concatenate((testData[:,:2], np.array(pred).reshape(len(pred), 1)),axis=1)
    return pred, mse

epochs = 30000
pred2, mse2 = trainTest(2, epochs)
pred8, mse8 = trainTest(8, epochs)
pred50, mse50 = trainTest(50, epochs)

testData = genGridData(9, f)

legendT = mpatches.Patch(color='black', label='target')
legend2 = mpatches.Patch(color='green', label='2 neurons')
legend8 = mpatches.Patch(color='red', label='8 neurons')
legend50 = mpatches.Patch(color='blue', label='50 neurons')
plt.legend(handles=[legendT, legend2, legend8, legend50])

pltData(testData, 'black')
pltData(pred2, 'green')
pltData(pred8, 'red')
pltData(pred50, 'blue')
print(mse2, mse8, mse50)
plt.show()