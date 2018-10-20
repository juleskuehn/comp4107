import numpy as np
import scipy.linalg as sp
import math
import matplotlib.pyplot as plt
import tensorflow as tf

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
trainZ = []
for xy in trainXY:
    trainZ.append(f(xy[0],xy[1]))

testXY = []
testCoords = genCoords(9)
for x in testCoords:
    for y in testCoords:
        testXY.append([x, y])
testZ = []
for xy in testXY:
    testZ.append(f(xy[0],xy[1]))


tf.logging.set_verbosity(tf.logging.ERROR)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_h2, w_o):
    # this is a basic mlp, think 2 stacked logistic regressions
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1))
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    # note that we dont take the softmax at the end because our cost fn does that for us
    return tf.matmul(h, w_o)


size_h1 = tf.constant(10, dtype=tf.int32)
size_h2 = tf.constant(6, dtype=tf.int32)

# Input: x and y
X = tf.placeholder("float", [None, 2])

# Output: z
Y = tf.placeholder("float", [None, 1])

# Input layer -> Hidden Layer 1
w_h1 = init_weights([2, size_h1])
# Hidden layer 1 -> Hidden Layer 2
w_h2 = init_weights([size_h1, size_h2])
# Hidden layer 2 -> Output layer
w_o = init_weights([size_h2, 1])


py_x = model(X, w_h1, w_h2, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=py_x, labels=Y))  # compute costs

train_op = tf.train.RMSPropOptimizer(
    0.05).minimize(cost)  # construct an optimizer

predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

batch_size = 10
# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()
    for i in range(3): # Epochs
        sess.run(train_op, feed_dict={
                X: trainXY, Y: np.array(trainZ).reshape(100,1)})
        predict = sess.run(py_x, feed_dict={X: testXY})
        print(i, np.mean(np.argmax(testXY, axis=1) ==
                         predict))
    saver.save(sess, "mlp/session.ckpt")
