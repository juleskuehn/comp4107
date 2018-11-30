# COMP 4107
# Fall 2018
# Assignment 4
# Yunkai Wang, student number 100968473
# Jules Kuehn, student number 100661464

"""
1. Modification of the dataset loaded to classify the CIFAR-10, 10 class image data.
2. Changing the number of convolutional layers and sizes of max pooling layers. You must investigate 5 different model scenarios.
3. Provide a PDF image of your computational graph. This must be captured from TensorBoard.
4. Provide a capability to view weight histograms using TensorBoard. You must be able to checkpoint your model during training. See tutorial for details.
5. Provide a chart of the accuracy of your network for 1-15 epochs for the scenarios investigated.
6. Provide the capability to show the top 9 patches. Examples are shown on slides 63-65 of the CNN notes.
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden):
    # 1. convolution layer
    # 2. max pool layer
    # 3. sigmoidal layer (or relu)
    # 4. soft max layer

    # Like in lecture slides, there is a "depth" to the number of hidden layers
    # A feature map is one of the "layers" in this tensor. Here there are 32.
    # Padding is to make it match with the window sizes
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
                        # What's with the 4 numbers? Don't worry about outside 1's
                        # The inner two digits are the ones that matter (how far you move)
                        # Just leave outer digits as ones.
                        # Kernel size below is 2 x 2 (as shown in slides)
                        # [_, x, y, _]
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
                        # End of functional convolutional unit (Functional layer + max pool)
    l1 = tf.nn.dropout(l1, p_keep_conv)

    # Transform from CNN to feed forward style
    l3 = tf.reshape(l1, [-1, w_fc.get_shape().as_list()[0]])    # reshape to (?, 14x14x32)
    # Now we have a nice "line" of neurons
    l3 = tf.nn.dropout(l3, p_keep_conv)
    # "Sigmoid" layer (here relu)
    l4 = tf.nn.relu(tf.matmul(l3, w_fc))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    # This will be the softmax layer (not shown here, but that's how code works, see below)
    pyx = tf.matmul(l4, w_o)
    return pyx


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

"""
Part 1: Load CIFAR-10
"""
from keras.datasets import cifar10
(trX, trY), (teX, teY) = cifar10.load_data()

trX = trX.reshape(-1, 32, 32, 3)  # 32x32x3 input img
teX = teX.reshape(-1, 32, 32, 3)  # 32x32x3 input img

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

# This should change to 3x3x3 (last 3 is for RGB input) ???
w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w_fc = init_weights([32 * 14 * 14, 625]) # FC 32 * 14 * 14 inputs, 625 outputs
w_o = init_weights([625, 10])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w_fc, w_o, p_keep_conv, p_keep_hidden)

# Note, this is the same as original MLP.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(100):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0})))
