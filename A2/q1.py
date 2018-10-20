"""
Notes:
- Validation data randomly sampled

Part A:

1. Using the GradientDescentOptimizer, investigate the performance of a 2 layer neural network with 2, 8 and 50 hidden layer neurons.
2. You must produce a contour diagram similar to Fig.3.
3. Include a table of MSE for the 3 different network sizes.


 two inputs (x, y), and one output (z = f(x,y)). 
two inputs, one layer of sigmoid transfer function neurons and one linear transfer function output neuron.

traingd
traingdm
traingrms

Part B:

1. Determine an appropriate number of epochs for convergence.
2. Plot the variation of MSE against epoch number for each of the 3 methods.
3. Plot a bar chart of the CPU time taken per epoch for each of the 3 methods.
4. Which method provides the best accuracy at the end of 100 epochs of training?
5. Which method is the most accurate when the training error is reached?


If the size of the network is too large it may run a risk of overfitting the training set and loses its generalization ability for unseen data.
early stopping: This technique monitors the error on a subset of the data (validation data) that does not actually take part in the training.
The training stops when the error on the validation data increases for a certain amount of iterations.

In order to examine the effect of early stopping on the training process, a randomly generated validation set is used during the trainrms training (Maximum validation failures=10, Erms=0.02 for the test set). 

Part C:

1. Confirm that approximately 8 neurons are a good choice for the current problem.
2. Run experiments across a range of hidden layer sizes and plot MSE at convergence against hidden layer size.
3. Reproduce (approximately) Fig. 6 and Fig. 7

"""

##########################
# GENERATE TRAINING DATA #
##########################

import numpy as np
import scipy.linalg as sp
import math
import matplotlib.pyplot as plt

# Tolerance
ε = 0.02 # MSE
MAX_EPOCHS = 100
LR = 0.02


def satisfied(fpx, fx, ε):
    """
    || f'(x) - f(x) ||< ε
    """
    return np.linalg.norm(fpx - fx) < ε


def f(x, y):
    """
    x, y ∈ [-1 1]
    """
    return math.cos(x + 6*0.35*y) + 2*0.35*x*y

def genCoords(n):
    coords = []
    step = 1 / ((n-1) / 2)
    x = -1
    while x <= 1:
        coords.append(x)
        x += step
    return coords

def f_prime(x, y):
    return

# Manipulate into format suitable for input to matrix
trainXY = []
trainCoords = genCoords(10)
for x in trainCoords:
    for y in trainCoords:
        trainXY.append([x, y])
trainXY = np.array(trainXY)
trainZ = []
for xy in trainXY:
    trainZ.append(f(xy[0],xy[1]))
trainZ = np.array(trainZ)

testXY = []
testCoords = genCoords(9)
for x in testCoords:
    for y in testCoords:
        testXY.append([x, y])
testXY = np.array(testXY)
testZ = []
for xy in testXY:
    testZ.append(f(xy[0],xy[1]))
testZ = np.array(testZ)

plt.contour(
        testXY[:, 0].reshape((9,9)),
        testXY[:, 1].reshape((9,9)),
        testZ.reshape((9,9)),
        10
    )


import tensorflow as tf

# Hide deprecation warnings
tf.logging.set_verbosity(tf.logging.ERROR)


def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))


def model(XY, size_h):
    # Weights into hidden layer (2 is because 2 inputs - x and y)
    w_h = init_weights([2, size_h], 'uniform') 
    # b_h = init_weights([2, size_h], 'zeros')
    # Weights into output layer (from hidden layer)
    # w_o = init_weights([size_h, 1], 'xavier', xavier_params=(size_h, 1))
    w_o = init_weights([size_h, 1], 'uniform')
    # b_o = init_weights([1, 1], 'zeros')

    h = tf.nn.sigmoid(tf.matmul(XY, w_h))
    # note that we dont take the softmax at the end because our cost fn does that for us
    return tf.matmul(h, w_o)


trainZ = trainZ.reshape(100,1)
testZ = testZ.reshape(81,1)

# Randomize order of training data
trainXYZ = np.concatenate((trainXY, trainZ), axis=1)
np.random.shuffle(trainXYZ)
trainXY = trainXYZ[:,:2]
trainZ = trainXYZ[:,2].reshape(100,1)

# Input layer: x and y
XY = tf.placeholder(tf.float32, [None, 2])
# Output layer: z
Z = tf.placeholder(tf.float32, [None, 1])

numNeurons = 8
# Initialize compute node
yhat = model(XY, numNeurons) # 2 hidden neurons


# # Training node
# train_op = tf.train.AdamOptimizer().minimize(
#     tf.nn.l2_loss(yhat - Z)
#     )

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=yhat, labels=Z))  # compute costs

train_op = tf.train.RMSPropOptimizer(
    0.05).minimize(cost)  # construct an optimizer

saver = tf.train.Saver()

# output = tf.nn.softmax(yhat)

MINI_BATCH_SIZE = 100
# Launch the graph in a session
with tf.Session() as sess:
    # sess.run(tf.initialize_all_variables())
    tf.global_variables_initializer().run()
    errors = []
    for i in range(51):
        for start, end in zip(range(0, len(trainXY)+1, MINI_BATCH_SIZE),
                    range(MINI_BATCH_SIZE, len(trainXY)+1, MINI_BATCH_SIZE)):
            sess.run(train_op, feed_dict={
                    XY: trainXY[start:end],
                    Z: trainZ[start:end]
                })
        mse = sess.run(tf.nn.l2_loss(yhat - trainZ),  feed_dict={XY: trainXY})
        # output = sess.run(Z, feed_dict={XY: testXY})
        errors.append(mse)
        if i%10 == 0:
            print(f"epoch {i}, validation MSE {mse}")
            print(sess.run(yhat, feed_dict={XY: testXY}))

    plt.plot(errors)
    plt.xlabel('#epochs')
    plt.ylabel('MSE')
    # print(prediction)
    # print(testZ)
    # for i in range(len(testZ)):
    #     print(f"Actual: {testZ[i]:3.3f}, Predict: {prediction[i]:3.3f}")
    saver.save(sess, "q1/session.ckpt")

