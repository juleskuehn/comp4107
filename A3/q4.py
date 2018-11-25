"""
Using Principal Component Analysis (PCA) and scikit-learn face data compare the classification accuracy of faces when using this orthonormal basis as input to a feed forward neural network when compared to using the raw data as input features. You are expected to document (a) the size of your feed forward neural network in both cases and (b) the prediction accuracy of your neural networks using a K-fold analysis. Experimental investigation over a range of network sizes and parameters is expected.
"""

import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# load data with/without PCA
def load_data():
    lfw_people = fetch_lfw_people(min_faces_per_person=5, resize=0.4)
    data = lfw_people.data
    names = lfw_people.target_names

    num_features = data.shape[1]
    num_classes = names.shape[0]
    labels = np.eye(num_classes)[lfw_people.target]

    # split data into training/testing sets
    x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=45)

    # use PCA to get orthonormal data
    pca = PCA()
    pca.fit(data)

    pca_x_train = pca.transform(x_train)
    pca_x_test = pca.transform(x_test)

    return x_train, x_test, y_train, y_test, pca_x_train, pca_x_test, num_features, num_classes

    # data = lfw_people.data
    # target = lfw_people.target

    # X_train, X_test, y_train, y_test = train_test_split(
    #     data, target, test_size=0.25, random_state=42)

    # if pca:
    #     data = PCA(n_components=50, svd_solver='randomized', whiten=True).fit(X_train)
    #     X_train_pca = data.transform(X_train)
    #     X_test_pca = data.transform(X_test)

    #     X_train = X_train_pca
    #     X_test = X_test_pca

    # return X_train, y_train, X_test, y_test, lfw_people.target_names.shape[0]


# initialize weight randomly
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# create the model
def model(X, w_h1, w_h2, w_o):
    h1 = tf.nn.sigmoid(tf.matmul(X, w_h1))
    h = tf.nn.sigmoid(tf.matmul(h1, w_h2))
    return tf.matmul(h, w_o)

# create the NN
def create_feed_forward_NN(h1_size, h2_size, output_size, input_size=1850, learning_rate=0.05):
    size_h1 = tf.constant(h1_size, dtype=tf.int32)
    size_h2 = tf.constant(h2_size, dtype=tf.int32)

    X = tf.placeholder("float", [None, input_size])
    Y = tf.placeholder("float", [None, output_size])

    w_h1 = init_weights([input_size, size_h1]) # create symbolic variables
    w_h2 = init_weights([size_h1, size_h2])
    w_o = init_weights([size_h2, output_size])

    py_x = model(X, w_h1, w_h2, w_o)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # compute costs
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # construct an optimizer
    predict_op = tf.argmax(py_x, 1)

    return X, Y, py_x, cost, train_op, predict_op

def cross_validation(X_train, y_train, X_test, y_test, train_op, predict_op, X, Y, num_epoch=100):
    with tf.Session() as sess:
        # you need to initialize all variables
        tf.global_variables_initializer().run()
        for _ in range(num_epoch):
            sess.run(train_op, feed_dict={X: X_train, Y: y_train})
        acc = np.mean(np.argmax(y_test, axis=1) == sess.run(predict_op, feed_dict={X: X_test}))
    return acc

def experiment(pca=False, h1_size=200, h2_size=200, learning_rate=0.05):
    x_train, x_test, y_train, y_test, pca_x_train, pca_x_test, input_size, output_size = load_data()

    X, Y, _, _, train_op, predict_op = create_feed_forward_NN(h1_size, h2_size, output_size=output_size,
                                input_size=input_size, learning_rate=learning_rate)

    if not pca:
        accuracies = cross_validation(x_train, y_train, x_test, y_test, train_op, predict_op, X, Y)
    else:
        accuracies = cross_validation(pca_x_train, y_train, pca_x_test, y_test, train_op, predict_op, X, Y)

    return accuracies

experiment(h1_size=200, h2_size=50)