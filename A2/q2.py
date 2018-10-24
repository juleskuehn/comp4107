"""
COMP4107 Fall 2018 Assignment 2
Yunkai Wang
100968473

35 inputs and 31 output neurons (see Fig.11). The network receives 35 Boolean values, which represents one character.
The logsig (Log Sigmoid) is chosen as the transfer function for both hidden and output layers.
softmax function on the output layer.

Part A

1. Run experiments with hidden neuron numbers in the range 5-25.
2. Plot a chart of recognition error against the number of hidden neurons.


- weights normally distributed around zero.

In the first step, the network is trained on the ideal data for zero decision errors (see Fig.13 (a)).
In the second step, the network is trained on noisy data as shown in Fig.10 for several passes (e.g.10 passes) for a proper performance goal (0.01 is used in the program).
In the final step, it is trained again on just ideal data for zero decision errors (see Fig.13 (b)).

Part B

1. Confirm that Fig.13 is a reasonable representation of performance for the optimal number of hidden layer neurons chosen.
2. Plot a chart in support of (1)


Part C

1. Create testing data that has between 0 and 3 bits noise.
2. Confirm that you can produce the recognition accuracy shown in Fig. 14.
"""

import tensorflow as tf
import numpy as np
from random import randint
import math

tf.logging.set_verbosity(tf.logging.ERROR)

# create array containing A to Z
characters = [chr(letter) for letter in range(65, 91)]

# add last five characters to the array
characters.extend(['j', 'i', 'n', 'y', 'u'])

# generate noise-free data
def genData():
    # creating input array
    # the input are stored in q2-patterns.txt, which basically convert the 7 * 5 pixels into a 7 * 5 array for each input character
    input = []
    with open('q2-patterns.txt', 'r') as f:

        # read all 62 images
        for _ in range(31):
            pixels = []
            # read the pixels that represent one image into 1-D array
            for i in range(7):
                line = list(f.readline())
                for j in range(5):
                    pixels.append(int(line[j]))
            f.readline()  # skip the empty line between images
            input.append(pixels)

    # create output array containing 1 - 31
    output = [[1 if i == counter + 1 else 0 for i in range(31)] for counter in range(31)]
    return input, output

# generate data with some errors
def genTestData(numInverseBit = 0):
    input, output = genData()

    for line in input:
        # reverse some of the pixels as 'noise'
        for _ in range(numInverseBit):
            pos = randint(0, 34)
            if line[pos] == 1:
                line[pos] = 0
            else:
                line[pos] = 1
    return input, output


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h1))
    return tf.matmul(h, w_o)

def degit_recognition(n_hidden = 15, noise = 0, lr = 0.05):
    tr_input, tr_output = genData()

    # generate training data with 3 bits of error
    tr_e_input, tr_e_output = genTestData(3)

    # Input: 35 pixels
    X = tf.placeholder("float", [None, 35])

    # Output: 31 characters
    Y = tf.placeholder("float", [None, 31])

    size_h1 = tf.constant(n_hidden, dtype=tf.int32)

    # Input layer -> Hidden Layer 1
    w_h1 = init_weights([35, size_h1])
    # Hidden layer 1 -> Output layer
    w_o = init_weights([size_h1, 31])

    py_x = model(X, w_h1, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.AdamOptimizer().minimize(cost) # construct an optimizer
    predict_op = tf.argmax(py_x, 1)

    costs = []

    with tf.Session() as sess:
        epoch = 0
        tf.global_variables_initializer().run()
        while True:
            epoch += 1
            _, c = sess.run([train_op, cost], feed_dict = {X: tr_input, Y: tr_output})
            # Show progress
            if epoch % 100 == 0:
                print("Cost: ", c)
                print(np.mean(np.argmax(tr_output, axis=1) ==
                                 sess.run(predict_op, feed_dict={X: tr_input})))
                print(sess.run(predict_op, feed_dict={X: tr_input}))
            if epoch % 10 == 0:
                costs.append(c)
            if epoch >= 1000:
                break

    print("Finished step 1 with ", epoch, " epochs")


    error = 0
    pred = pred[0]
    for i in range(len(pred)):
        x = pred[i]
        max = -math.inf
        pos = 0
        for j in range(len(x)):
            if x[j] > max:
                max = x[j]
                pos = j

        y = te_output[i]
        max = -math.inf
        y_pos = 0
        for j in range(len(y)):
            if x[j] > max:
                max = y[j]
                pos = j

        if not pos == y_pos:
            error += 1

    print("Number of incorrect prediction: ", error)
    print("Error rate: ", error / len(pred))

    # Reshape prediction into format of testData for comparison [[x, y, z]]
    pred = np.concatenate((testData[:,:35], np.array(pred).reshape(len(pred), 1)),axis=1)
    return pred, mse, costs
pred, mse, costs = degit_recognition()
print(pred, mse, costs)
