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

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import sample
import math

VERBOSE = False

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
    output = np.array([counter for counter in range(31)]).reshape(31, 1)

    return np.concatenate((input, output), axis=1)

# generate data with some errors
def genTestData(numInverseBit = 0):
    input = genData()

    for line in input:
        # reverse some of the pixels as 'noise'
        positions = sample(range(0, 34), numInverseBit)

        for pos in positions:
            if line[pos] == 1:
                line[pos] = 0
            else:
                line[pos] = 1

    return input


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, b1, w_o, b2):
    h = tf.nn.sigmoid(tf.matmul(X, w_h1) + b1)
    return tf.matmul(h, w_o) + b2

def train(sess, t_op, p_op, cost, X, Y, input, output, error_rate = 0):
    accuracy = 0
    epoch = 0
    accuracies = []
    costs = []
    while not accuracy >= 1 - error_rate:
        epoch += 1
        _, c = sess.run([t_op, cost], feed_dict = {X: input, Y: output})
        # Show progress
        if epoch % 10 == 0 and VERBOSE:
            print("epoch: ", epoch, ", accurage:", np.mean(np.argmax(output, axis=1) ==
                             sess.run(p_op, feed_dict={X: input})))
        accuracy = np.mean(np.argmax(output, axis=1) ==
                         sess.run(p_op, feed_dict={X: input}))
        costs.append(c)
        accuracies.append(accuracy)

    return epoch, accuracies, costs

def digit_recognition(n_hidden = 15, training_noise = 3, lr = 0.01):
    # noise-free data
    tr_data = genData()
    np.random.shuffle(tr_data)
    tr_input = tr_data[:, :35]
    output = tr_data[:, 35]
    tr_output = [[1 if i == j else 0 for j in range(31)] for i in output]

    # generate noisy training data with the given noise level
    tr_e_data = genTestData(training_noise)
    np.random.shuffle(tr_e_data)
    tr_e_input = tr_e_data[:, :35]
    e_output = tr_e_data[:, 35]
    tr_e_output = [[1 if i == j else 0 for j in range(31)] for i in e_output]

    # Input: 35 pixels
    X = tf.placeholder("float", [None, 35])

    # Output: 31 characters
    Y = tf.placeholder("float", [None, 31])

    size_h1 = tf.constant(n_hidden, dtype=tf.int32)

    # Input layer -> Hidden Layer 1
    w_h1 = init_weights([35, size_h1])
    # Hidden layer 1 -> Output layer
    w_o = init_weights([size_h1, 31])

    # bias
    b1 = tf.Variable(tf.random_normal([size_h1], stddev=0.01))
    b2 = tf.Variable(tf.random_normal([31], stddev=0.01))

    py_x = model(X, w_h1, b1, w_o, b2)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.AdamOptimizer(lr).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(py_x, 1)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # step 1: training with noise-free data
        # training until accuracy goes to 100%
        epoch1, step1_accuracies, _ = train(sess, train_op, predict_op, cost, X, Y, tr_input, tr_output)

        if VERBOSE:
            print("Finished step 1 with ", epoch, " epochs")

        # step 2: training with noisy data
        # training until accuracy goes to 99%
        epoch2, _, _ = train(sess, train_op, predict_op, cost, X, Y, tr_e_input, tr_e_output, 0.01)

        if VERBOSE:
            print("Finished step 2 with ", epoch, " epochs")

        # step 3: training with noise-free data
        # training until accuracy goes to 100%
        epoch3, step3_accuracies, _ = train(sess, train_op, predict_op, cost, X, Y, tr_input, tr_output)

        if VERBOSE:
            print("Finished step 3 with ", epoch, " epochs")

        accuracies = []

        # testing NN with noisy data
        for i in range(4):
            t_acc = 0 # total accuracy
            # run this testing process for 10 times
            for _ in range(10):
                # generate noisy testing data
                te_data = genTestData(i)
                np.random.shuffle(te_data)
                te_input = te_data[:, :35]
                e_output = te_data[:, 35]
                te_output = [[1 if i == j else 0 for j in range(31)] for i in e_output]

                accuracy = np.mean(np.argmax(te_output, axis=1) == sess.run(predict_op, feed_dict={X: te_input}))
                t_acc += accuracy
            if VERBOSE:
                print("Noise level: ", i, ", accuracy: ", t_acc / 10)
            accuracies.append(t_acc / 10)
    return accuracies, epoch1, step1_accuracies, epoch3, step3_accuracies

# part a
def partA():
    noises = [i for i in range(4)]

    print("--------Running PART A------------")
    colors = ['red', 'blue', 'green', 'black', 'purple']
    results = []
    for n_hidden in range(5, 26, 5):
        t_acc = [0 for _ in range(4)]

        # for each number of hidden neurons, run the experiment 10 times so that the result is 'accurate'
        for _ in range(10):
            accuracies, _, _, _, _ = digit_recognition(n_hidden)
            for i in range(4):
                t_acc[i] += accuracies[i]

        # devide the accuracy by 10 since we added the sum of accuracy for 10 times, 1 - accuracy so that we get the error rate, * 100 so that it's a percentage
        for i in range(4):
            t_acc[i] = (1 - t_acc[i] / 10) * 100
        results.append(t_acc)
        print("Number of hidden neuron: ", n_hidden, ", accuracies: ", t_acc)

    plt.figure(1)
    plt.xlabel('Noise level')
    plt.ylabel('Percentage of recognition error')
    for i in range(len(results)):
        plt.plot(noises, results[i], colors[i])

    plt.show()

# as can be shown in the first firgure generated, when number of hidden neurons is 15, it's the optimal one, so for part b and c, we will be using 15 as the number of hidden neurons
n_hidden = 15

# part b
def partB():
    print("--------Running PART B------------")
    _, epoch1, step1_accuracies, epoch3, step3_accuracies = digit_recognition(n_hidden)
    step1_error = [1 - acc for acc in step1_accuracies]
    step3_error = [1 - acc for acc in step3_accuracies]
    plt.figure(2)
    plt.xlabel('Epoch number')
    plt.xticks(np.arange(0, max(step1_error), step=0.01))
    plt.yticks(np.arange(0, epoch1, step=10))
    plt.ylabel('Percentage of recognition error')
    plt.plot([count for count in range(0, epoch1)], step1_error, 'black')

    plt.figure(3)
    plt.xlabel('Epoch number')
    plt.ylabel('Percentage of recognition error')
    plt.plot([count for count in range(0, epoch3)], step3_error, 'black')

    plt.show()

# part c
def partC():
    print("--------Running PART C------------")
    results = []
    t_acc = [0 for _ in range(4)] # total accuracy trained with noise-free data only
    t_acc_noisy = [0 for _ in range(4)] # total accuracy trained with noisy data
    for _ in range(10):
        accuracies, _, _, _, _ = digit_recognition(n_hidden, 0)
        for i in range(4):
            t_acc[i] += accuracies[i]

        accuracies, _, _, _, _ = digit_recognition(n_hidden)
        for i in range(4):
            t_acc_noisy[i] += accuracies[i]

    for i in range(4):
        t_acc[i] = (1 - t_acc[i] / 10) * 100
        t_acc_noisy[i] = (1 - t_acc_noisy[i] / 10) * 100

    print("Accuracy for training with noise-free data: ", t_acc)
    print("Accuracy for training with noisy data: ", t_acc_noisy)

    plt.figure(4)
    plt.yticks(np.arange(0, 20, step=2))
    plt.xlabel('Noise level')
    plt.ylabel('Percentage of recognition error')
    plt.plot(noises, t_acc, 'b--')
    plt.plot(noises, t_acc_noisy, 'r')

def experiment():
    # partA()
    # partB()
    # partC()
    pass

experiment()
