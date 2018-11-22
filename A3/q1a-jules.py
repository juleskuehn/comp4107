# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn


"""
Question 1

(a) Using the scikit-learn utilities to load the MNIST data, implement a
Hopfield network that can classify the image data for a subset of the
handwritten digits.

Here, correct classification means that if we present an image of a '1' an
image of a '1' will be recovered; however, it may not be the original image
owing to the degenerate property of this type of network.

You are expected to document classification accuracy as a function of the
number of images used to train the network. Remember, a Hopfield network can
only store approximately 0.15N patterns with the "one shot" learning described
in the lecture (slides 58-74).
"""

# Part A: Subsample the data to only include images of '1' and '5'.

import numpy as np
import scipy.linalg as sp
import sympy
import os
import struct
from matplotlib import pyplot
import matplotlib as mpl
import math
import random
from q1a_functions import get_1_and_5
from q1a_functions import print_num


all_data = get_1_and_5()

print_num(all_data, 1, 0)
print_num(all_data, 5, 0)

data = np.concatenate((all_data[1], all_data[5]), axis=1).transpose()
numSamples = 100
sample_indices = random.sample(list(range(len(data))), numSamples)

def train_weights(train_data):
    p = len(train_data)
    W = np.zeros((p, p))
    
    for x in train_data:
        x = np.array(
            [1 if e > 127 else -1 for e in x]
        )
        
        W += x.dot(x.transpose())
        # print(W)
    W -= np.dot(np.identity(p), p)
    return W

w = train_weights(data[sample_indices])