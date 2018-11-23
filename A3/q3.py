
'''
COMP4107 fall 2018 assignment 3 question 3
Student name: Yunkai Wang
Student num: 100968473

We can use self organizing maps as a substitute for K-means.

In Question 2, K-means was used to compute the number of hidden layer neurons to be used in an RBF network. Using a 2D self-organizing map compare the clusters when compared to K-means for the MNIST data. Sample the data to include only images of '1' and '5'. Use the scikit-learn utilities to load the data. You are expected to (a) document the dimensions of the SOM computed and the learning parameters used to generate it (b) provide 2D plots of the regions for '1' and '5' for both the SOM and K-means solutions. You may project your K-means data using SVD to 2 dimensions for display purposes.
'''

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, sqrt
from random import shuffle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# minisom library found at https://github.com/JustGlowing/minisom
# Can be installed by running 'pip3 minisom'
from minisom import MiniSom

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# return part of the data that matches the given label
def partition(x_data, y_data,label):
    return [(np.array(input).ravel(), label) for (input, output) in zip(x_data, y_data) if output == label]


def getTrainingData():
    # get the datas for 1 and 5
    ones = partition(x_train, y_train, 1)
    fives = partition(x_train, y_train, 5)
    # shuffle the training data so that the execution process won't always be the same
    trainingData = ones + fives
    shuffle(trainingData)

    return trainingData


def getTestingData():
    # get the datas for 1 and 5
    ones = partition(x_test, y_test, 1)
    fives = partition(x_test, y_test, 5)
    # shuffle the training data so that the execution process won't always be the same
    testingData = ones + fives
    shuffle(testingData)

    return testingData


# calculate the number of output neurons, n is the number of samples,
# given the number of samples n, there should be >= 5 * sqrt(n) neurons
def calculate_dimension(n):
    return ceil(sqrt(5 * sqrt(n)))


def draw(som, x, y, data, title):
    for item in data:
        pixels, label = item
        i, j = som.winner(pixels)
        plt.text(i, j, label)

    plt.axis([0, x, 0, y])
    plt.title(title)
    plt.savefig("./document/" + title + ".png")
    plt.clf() # clear current figure for the next figure
    # plt.show()


def train(trainingData, num_epoch=500, num_split=2, input_len=784, sigma=1, learning_rate=0.1):
    x = calculate_dimension(len(trainingData))
    y = calculate_dimension(len(trainingData))

    # create the self organizing map
    som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate)

    data = [pixels for pixels, _ in trainingData]

    # draw the SOM before training
    draw(som, x, y, trainingData, "Before training")

    for i in range(num_split):
        # training the SOM for the number of epochs specified
        som.train_random(data, int(num_epoch / num_split))

        # draw the SOM after training
        draw(som, x, y, trainingData, 'After training for %s epochs' % int((i + 1) * (num_epoch / num_split)))
    
    return som


def display_PCA(trainingData):
    # unpack the pixels and the labels
    data = [pixels for pixels, _ in trainingData]
    labels = [label for _, label in trainingData]

    # create PCA for 1 and 5
    pca = PCA(n_components=2)
    pca.fit(data)
    pca_list = pca.transform(data) # reduce the data to 2D for displaying purpose
    plt.scatter(pca_list[:, 0], pca_list[:, 1], c=labels, s=0.5)
    title = "Without kmeans"
    plt.title(title)
    plt.savefig("./document/" + title + ".png")
    plt.clf() # clear current figure for the next plot

    num_clusters = range(2, 6)
    for cluster_size in num_clusters:
        kmeans = KMeans(n_clusters=cluster_size)
        kmeans.fit(data)
        title = "With " + str(cluster_size) + " clusters"
        plt.scatter(pca_list[:, 0], pca_list[:, 1], c=kmeans.labels_, s=0.5)
        plt.title(title)
        plt.savefig("./document/" + title + ".png")
        plt.clf() # clear current figure for the next plot
        # plt.show()


# use only 1024 data for just for a faster running time
trainingData = getTrainingData()[:1024] # get the training data

input_len = 784 # each image in MNIST database contains 784 pixels
sigma = 1 # not sure what to choose for now, just testing 1
learning_rate = 0.1 # not sure what to choose for now, just testing 0.1

# som = train(trainingData, num_epoch=500, learning_rate=learning_rate, sigma=sigma, input_len=input_len)
# plt.show()

display_PCA(trainingData)
