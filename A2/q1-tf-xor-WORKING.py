import numpy as np
import scipy.linalg as sp
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import random

# Tolerance
ε = 0.02 # MSE
MAX_EPOCHS = 100
LR = 0.02

def satisfied(fpx, fx, ε):
    return np.linalg.norm(fpx - fx) < ε


def f(x, y):
    return math.cos(x + 6*0.35*y) + 2*0.35*x*y       # x, y ∈ [-1 1]

def genCoords(n):
    coords = []
    step = 1 / ((n-1) / 2)
    x = -1
    while x <= 1:
        coords.append(x)
        x += step
    return coords

# Manipulate into format suitable for input to matrix
trainXY = []
trainCoords = genCoords(10)
for x in trainCoords:
    for y in trainCoords:
        trainXY.append([x, y])
trainXY = np.array(trainXY)
trainZ = []
for xy in trainXY:
    trainZ.append((f(xy[0],xy[1]) + 1) / 2)
trainZ = np.array(trainZ).reshape(100,1)

testXY = []
testCoords = genCoords(9)
for x in testCoords:
    for y in testCoords:
        testXY.append([x, y])
testXY = np.array(testXY)
testZ = []
for xy in testXY:
    testZ.append((f(xy[0],xy[1]) + 1) / 2)
testZ = np.array(testZ)

plt.contour(
        trainXY[:, 0].reshape((10,10)),
        trainXY[:, 1].reshape((10,10)),
        trainZ.reshape((10,10)),
        10
    )

# Generate validation data
validationXY = [(random.random()*2 - 1, random.random()*2 - 1) for _ in range(100)]
validationZ = [(f(x, y)+1) / 2 for x, y in validationXY]

# Randomize order of training data
trainXYZ = np.concatenate((trainXY, trainZ), axis=1)
np.random.shuffle(trainXYZ)
trainXY = trainXYZ[:,:2]
trainZ = trainXYZ[:,2].reshape(100,1)

# Dataset
x_data = trainXY
y_data = trainZ

# Hyperparamters
n_input = 2
n_hidden = 50
n_output = 1
lr = 0.5
epochs = 30000
display_step = 100

# Placeholders
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Weights
W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# Bias
b1 = tf.Variable(tf.zeros([n_hidden]))
b2 = tf.Variable(tf.zeros([n_output]))

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hy = tf.sigmoid(tf.matmul(L2, W2) + b2)
cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y) * tf.log(1-hy))
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(epochs):
        _, c = sess.run([optimizer, cost], feed_dict = {X: x_data, Y: y_data})
        # if satisfied(sess.run([hy], feed_dict = {X: testXY}), testZ, ε):
        #     break
        if c < 0.01:
            break
        if step % display_step == 0:
            print("Cost: ", c)
        

    pred = sess.run([hy], feed_dict = {X: testXY})
    print(pred)

for x, y, f in zip(testXY, testZ, pred[0]):
    print(f"{x[0]:5},{x[1]:5}: {y:.3f} vs {f[0]:.3f}")

# plt.contour(
#         trainXY[:, 0].reshape((10,10)),
#         trainXY[:, 1].reshape((10,10)),
#         (trainZ*2 - 1).reshape((10,10)),
#         10
#     )

plt.contour(
        testXY[:, 0].reshape((9,9)),
        testXY[:, 1].reshape((9,9)),
        (np.array([x[0] for x in pred[0]])*2 - 1).reshape((9,9)),
        10
    )

# plt.contour(
#         trainXY[:, 0].reshape((10,10)),
#         trainXY[:, 1].reshape((10,10)),
#         (np.array([x[0] for x in pred[0]])*2 - 1).reshape((10,10)),
#         10
#     )

plt.show()