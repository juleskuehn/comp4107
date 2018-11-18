# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn

# k-fold code based on https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

import numpy as np
import tensorflow as tf
import random
import math
from q2_kmeans import k_means
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import StratifiedKFold
from q2_rbflayer import RBFLayer, InitCentersRandom

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Prepare the data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train / 255.0, x_test / 255.0, axis=0).reshape(70000, 28 * 28)
y = np.append(y_train, y_test, axis=0)

# Set random seed for better reproducibility
seed = 7
random.seed(seed)
np.random.seed(seed)


def rbf_activation(input_vectors, centers, betas):
    out_vectors = []
    for i in range(len(centers)):
        center = centers[i]
        beta = betas[i]
        out_vectors.append([])
        for input_vector in input_vectors:
            diff = np.linalg.norm(input_vector - center, ord=1)
            out_vectors[i].append(math.exp(-beta * diff**2))
    return np.array(out_vectors).transpose()


def train_test_kfold(x, y, num_folds=5, num_centers=200,
                     epochs=100, dropout=0, lr=0.01):
    print("Begin training and testing:")
    print(len(x), "samples,", num_centers, "centers,", num_folds, "folds,",
            lr, "learning rate,", dropout, "dropout,", epochs, "epochs.")
    
    # This algorithm attempts to balance the number of instances of each class
    # (neither splitting randomly, or sequentially)
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=False)
    cvscores = []

    for train, test in kfold.split(x, y):

        print("Running k-means for training iteration", len(cvscores)+1)
        # Prevent slow k_means by limiting number of samples
        k_means_samples = min(len(x), 1000)

        # Initializing centers randomly within the range of each dimension
        # gives very poor results; instead sampling random training points
        centers, betas = k_means(x[train][:k_means_samples], num_centers,
                              max_epochs=100, sample_centers=True, seed=seed)
        
        # Create model with input, dropout, and output layer
        model = Sequential()
        # Input layer is RBF. Can't figure out how to write as a Keras layer
        # So I manually process the input vectors through RBF. Now there are
        # num_centers inputs to the Dropout layer.
        print("Running RBF activations on", len(x), "input vectors")
        x_train_rbf = rbf_activation(x[train], centers, betas)
        x_test_rbf = rbf_activation(x[test], centers, betas)
        # x_train_rbf = x[train]
        # x_test_rbf = x[test]
        model.add(Dropout(dropout))
        # Output layer is linear activation, one bit hot
        model.add(Dense(10, activation=tf.nn.softmax))

        # Compile model
        optimizer = tf.keras.optimizers.Adam(lr=lr)
        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=optimizer, metrics=['accuracy'])
        
        # Fit the model
        print("Training the model for", epochs, "epochs at learning rate", lr)
        model.fit(x_train_rbf, y[train], epochs=epochs, verbose=0)

        # Evaluate model
        scores = model.evaluate(x_test_rbf, y[test], verbose=1)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)

    print(len(x), "samples,", num_centers, "centers,", num_folds, "folds,",
            lr, "learning rate,", dropout, "dropout,", epochs, "epochs.")
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    print("\n")
    return np.mean(cvscores)


def test_hidden_layer_size():
    # Results for different sizes of hidden layer
    results = {}

    numSamples = 1000
    sampleIndices = random.sample(list(range(len(x))), numSamples)

    for k in [10, 20, 40, 80, 160, 320, 640]:
        results[k] = train_test_kfold(
            x[sampleIndices], y[sampleIndices],
            num_folds=3, num_centers=k, epochs=100, dropout=0, lr=0.05)

    print(results)
    return results
    """
    {
        10: 71.02465320827038,
        40: 82.60398837033368,
        80: 84.31194848736659,
        160: 86.4130433253927,
        320: 84.01040230348342,
        640: 42.5716640874277
    }
    """


def test_dropout():
    # Results for different dropout levels
    results = {}

    numSamples = 5000
    sampleIndices = random.sample(list(range(len(x))), numSamples)
    for p in [0, 0.1, 0.2, 0.4, 0.6, 0.8]:
        results[p] = train_test_kfold(
            x[sampleIndices], y[sampleIndices],
            num_folds=3, num_centers=320, epochs=500, dropout=p, lr=0.01)
    print(results)
    return results

    """
    for 160 centers, 1000 samples, and 100 epochs, lr=0.05:
    {
        0: 86.49364913138595,
        0.1: 86.49095127939911,
        0.2: 84.80501058292278,
        0.4: 83.68496464162325,
        0.6: 80.80279256419999,
        0.8: 77.10251759336823
    }
    for 320 centers, 5000 samples, and 500 epochs, lr=0.01:
    {
        0: 92.25957215181523,
        0.1: 92.41908419990857,
        0.2: 91.77876615314769,
        0.4: 90.65975656015401,
        0.6: 88.81949583464971,
        0.8: 85.61887915837953
    }
    """