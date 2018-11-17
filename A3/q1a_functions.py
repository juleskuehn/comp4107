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

# read() function from
# https://gist.github.com/akesling/5358964
# All other code written by Jules


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    def get_img(idx): return (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def get_training_1_and_5(training_data):
    """
    Format training data into desired form:
    a list A, of matrices Aj with shape (2**n, m)
    for each digit, where j = the digit
    """
    A = [[] for _ in range(10)]
    for image_tuple in training_data:
        label, pixels = image_tuple
        # Stack pixels into single vector
        if label == 1 or label == 5:
            A[label].append((np.ndarray.flatten(pixels)))
    return [np.array(Aj, dtype="uint8").transpose() for Aj in A]


def num_to_2D_list(data, digit, index):
    return list((list(row) for row in list(
        data[digit][:, index].transpose().reshape(28, 28))))


def print_num(data, digit, index):
    a = num_to_2D_list(data, digit, index)
    for row in a:
        row_str = ''
        for char in row:
            if char == 0:
                row_str += ' '
            elif char < 128:
                row_str += ':'
            else:
                row_str += '#'
        print(row_str)
    print('-' * 28)


def get_1_and_5():
    return get_training_1_and_5(list(read("training"))+list(read("testing")))

