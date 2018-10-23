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


tf.logging.set_verbosity(tf.logging.ERROR)

# Input: 35 pixels
X = tf.placeholder("float", [None, 35])

# Output: 31 characters
Y = tf.placeholder("float", [None, 31])

# creating input array
# the input are stored in q2-patterns.txt, which basically convert the 7 * 5 pixels into a 7 * 5 array for each input character
input = []
with open('q2-patterns.txt', 'r') as f:

    # read all 62 images
    for _ in range(62):
        pixels = []
        # read the pixels that represent one image into 1-D array
        for i in range(7):
            line = list(f.readline())
            for j in range(5):
                pixels.append(int(line[j]))
        f.readline()  # skip the empty line between images
        input.append(pixels)


# create output array containing A to Z
output = [chr(letter) for letter in range(65, 91)]

# add last five characters to the output array
output.extend(['j', 'i', 'n', 'y', 'u'])

def degit_recognition(hidden):

    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(X, w_h1, w_o):
        h = tf.nn.sigmoid(tf.matmul(X, w_h1))
        return tf.matmul(h, w_o)

    size_h1 = tf.constant(hidden, dtype=tf.int32)

    # Input layer -> Hidden Layer 1
    w_h1 = init_weights([35, size_h1])
    # Hidden layer 1 -> Output layer
    w_o = init_weights([size_h2, 31])

    py_x = model(X, w_h1, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=py_x, labels=Y))  # compute costs

    train_op = tf.train.RMSPropOptimizer(
        0.05).minimize(cost)  # construct an optimizer

    predict_op = tf.argmax(py_x, 1)

    saver = tf.train.Saver()

    # Launch the graph in a session
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        print(range(0, len(trX), 128))
        for i in range(3): # Epochs
            for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)): # Stochastic gradient descent (in small batches, backprop)
                # print((start, end))
                sess.run(train_op, feed_dict={
                         X: trX[start:end], Y: trY[start:end]})
            predict = sess.run(py_x, feed_dict={X: teX})
            print(i, np.mean(np.argmax(teY, axis=1) ==
                             predict))
        saver.save(sess, "mlp/session.ckpt")
