import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.ERROR)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h1, w_h2, w_o):
    # this is a basic mlp, think 2 stacked logistic regressions
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1))
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    # note that we dont take the softmax at the end because our cost fn does that for us
    return tf.matmul(h, w_o)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

size_h1 = tf.constant(625, dtype=tf.int32)
size_h2 = tf.constant(300, dtype=tf.int32)

# Input: 784 pixels
X = tf.placeholder("float", [None, 784])

# Output: 10 digits (one bit hot)
Y = tf.placeholder("float", [None, 10])

# Input layer -> Hidden Layer 1
w_h1 = init_weights([784, size_h1])
# Hidden layer 1 -> Hidden Layer 2
w_h2 = init_weights([size_h1, size_h2])
# Hidden layer 2 -> Output layer
w_o = init_weights([size_h2, 10])


py_x = model(X, w_h1, w_h2, w_o)

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
