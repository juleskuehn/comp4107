# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn

# k-fold code based on https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
# RBF activation based on https://github.com/PetraVidnerova/rbf_keras/


import numpy as np
import tensorflow as tf
import random
import math
from q2_kmeans import k_means
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import StratifiedKFold
from q2_rbflayer import RBFLayer, InitCentersRandom


# Prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train / 255.0, x_test / 255.0, axis=0).reshape(70000, 28 * 28)
y = np.append(y_train, y_test, axis=0)

# Set random seed for reproducibility
seed = 7
np.random.seed(seed)


def rbf_activation(input_vectors, centers, betas):
    # C = K.expand_dims(tf.convert_to_tensor(centers, dtype='float32'))
    # H = C-K.transpose(input_vectors)
    # v = K.exp(-1 * betas * K.sum(H**2, axis=1))
    # print(v)
    # return v
    out_vectors = []
    for i in range(len(centers)):
        center = centers[i]
        beta = betas[i]
        out_vectors.append([])
        for input_vector in input_vectors:
            diff = np.linalg.norm(input_vector - center, ord=1)
            out_vectors[i].append(math.exp(-beta * diff**2))
    return np.array(out_vectors).transpose()


def train_test_kfold(x, y, folds, num_centers, dropout=0):
    # This algorithm attempts to balance the number of instances of each class
    # (neither splitting randomly, or sequentially)
    kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(x, y):
        # Run k-means on training data
        centers, betas = k_means(x[train], num_centers, max_epochs=500,
                                sample_centers=True)
        print(betas)
        # Create model
        model = Sequential()
        # Input layer is RBF. Can't figure out how to write as a Keras layer
        # So I manually process the input vectors through RBF. Now there are
        # num_centers inputs to the Dropout layer.
        x_train_rbf = rbf_activation(x[train], centers, betas)
        x_test_rbf = rbf_activation(x[test], centers, betas)
        # x_train_rbf = x[train]
        # x_test_rbf = x[test]
        model.add(Dropout(dropout))
        # Output layer is linear activation, one bit hot
        model.add(Dense(10, activation=tf.nn.softmax))
        # Compile model
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
        # Fit the model
        model.fit(x_train_rbf, y[train], epochs=100, verbose=1)
        # evaluate the model. Should I recalculate centers, betas?
        scores = model.evaluate(x_test_rbf, y[test], verbose=1)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

train_test_kfold(x[:1000], y[:1000], 3, 50, dropout=0)