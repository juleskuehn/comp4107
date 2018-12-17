# an experiment to see how well the clasification works using the feed-forward NN

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.model_selection import train_test_split
from data_helper import load_vectorized_data, vectorize_vector

batch_size = 128
test_size = 256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1)) # this is a basic mlp, think 2 stacked logistic regressions
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

trainingData = load_vectorized_data()
word_vectors = [np.array(word_vector) for word_vector, _ in trainingData]
categories = [category for _, category in trainingData]
categories = vectorize_vector(categories)
trX, teX, trY, teY = train_test_split(word_vectors, categories, test_size=0.1)

size_h1 = tf.constant(625, dtype=tf.int32)
size_h2 = tf.constant(300, dtype=tf.int32)

X = tf.placeholder("float", [None, 96])
Y = tf.placeholder("float", [None, 32])

w_h1 = init_weights([96, size_h1]) # create symbolic variables
w_h2 = init_weights([size_h1, size_h2])
w_o = init_weights([size_h2, 32])

py_x = model(X, w_h1, w_h2, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    for i in range(25):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))