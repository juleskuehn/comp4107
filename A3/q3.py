
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
from minisom import MiniSom


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# return part of the data that matches the given label
def partition(label):
    train_data = [(np.array(input).ravel(), label) for (input, output) in zip(x_train, y_train) if output == label]
    test_data = [(np.array(input).ravel(), label) for (input, output) in zip(x_test, y_test) if output == label]
    return train_data + test_data

# get the datas for 1 and 5
ones = partition(1)
fives = partition(5)

# shuffle the training data so that the execution process won't always be the same
trainingData = ones + fives
shuffle(trainingData)

# The grid should contains > 5 * sqrt(n) neurons, where n is the number of samples,
# there are 14190 data, 5 * sqrt(n) = 595, so we pick 25 * 25 neurons
x = 25
y = 25

input_len = 784 # each image in MNIST database contains 784 pixels
sigma = 1 # not sure what to choose for now, just testing 1
learning_rate = 0.1 # not sure what to choose for now, just testing 0.1


def draw(som, x, y, data, title):
    for item in data:
        pixels, label = item
        i, j = som.winner(pixels) # reshape the pixels array
        plt.text(i, j, label)
    plt.axis([0, x, 0, y])
    plt.title(title)
    plt.show()

def train(trainingData, x, y, num_epoch=100, input_len=784, sigma=1, learning_rate=0.1):
    # create the self organizing map
    som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate)

    # draw the SOM before training
    draw(som, x, y, trainingData, "Before training")

    # training the SOM for the number of epochs specified
    som.train_random([pixels for pixels, label in trainingData], num_epoch)

    # draw the SOM after training
    draw(som, x, y, trainingData, 'After training for %s epochs' % num_epoch)

train(trainingData, x, y, learning_rate=0.25)