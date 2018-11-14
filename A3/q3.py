
from random import shuffle

# load MNIST database
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original', data_home='.cache')

# return part of the data that matches the given label, used to get the data for digits 1 and 5
def partition(label):
    target = mnist.target.tolist()
    return [(input, label) for input in mnist.data[target.index(label) :
        target.index(label + 1) if label < 9 else len(target)]]

ones = partition(1)
fives = partition(5)

# shuffle the training data so that the execution process won't always be the same
trainingData = ones + fives
shuffle(trainingData)

