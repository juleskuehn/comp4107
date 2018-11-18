
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
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

# The grid should contains > 5 * sqrt(n) neurons, where n is the number of samples,
# there are 14190 data, 5 * sqrt(n) = 595, so we pick 25 * 25 neurons
x = 25
y = 25

input_len = 784 # each image in MNIST database contains 784 pixels
sigma = 1 # not sure what to choose for now, just testing 1
learning_rate = 0.1 # not sure what to choose for now, just testing 0.1

def draw(som, x, y, data, title, figure_num = 1):
    plt.figure(figure_num)
    for item in data:
        pixels, label = item
        i, j = som.winner(pixels)
        plt.text(i, j, label)
    plt.axis([0, x, 0, y])
    plt.title(title)
    plt.show()

def train(trainingData, x, y, num_epoch=500, num_split=2, input_len=784, sigma=1, learning_rate=0.1):
    figure_count = 1

    # create the self organizing map
    som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate)

    data = [pixels for pixels, label in trainingData]

    # draw the SOM before training
    draw(som, x, y, trainingData, "Before training", figure_count)
    figure_count += 1

    for i in range(num_split):
        # training the SOM for the number of epochs specified
        som.train_random(data, int(num_epoch / num_split))

        # draw the SOM after training
        draw(som, x, y, trainingData, 'After training for %s epochs' % int((i + 1) * (num_epoch / num_split)), figure_count)

        figure_count += 1
    
    return som

# def check_accuracy(som, testingData):
#     for pixels, label in testingData:



som = train(getTestingData(), x, y, num_epoch=500, learning_rate=0.25)
