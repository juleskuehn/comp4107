# COMP 4107, Fall 2018
# Assignment 1, Question 6
#
# Jules Kuehn
# 100661464

import numpy as np
import scipy.linalg as sp
import sympy
import os
import struct
from matplotlib import pyplot
import matplotlib as mpl
import math

"""
Code for reading the MNIST set (read() and show() functions) from:
https://gist.github.com/akesling/5358964
"""

def read(dataset = "training", path = "."):
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

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=pyplot.get_cmap('Greys'))
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()

# Jules' code begins here

# MISC (debugging) FUNCTIONS

# Reconstruct a 2D numpy uint8 array of 28x28
# from a vector of length 28**2
# (So it can be displayed as an image)
def reconstruct(Aa):
    tmp = []
    for i in range(0, 28):
        tmp.append([])
        for j in range(0, 28):
            tmp[i].append(Aa[i*28 + j])
    return np.array(tmp, dtype=np.uint8)

# How many singular values contain desired proportion of energy?
def bestRank(s, proportionEnergy):
    sumS = np.sum(s)
    k = 0
    e = 0
    while e < sumS * proportionEnergy :
        e += s[k]
        k += 1
    return k

# REQUIRED FUNCTIONS

# Format training data into desired form:
# a list A, of matrices Aj with shape (2**n, m)
# for each digit, where j = the digit
def getTrainingA(training_data):
    images = [
        [], [], [], [], [], [], [], [], [], []
    ]
    for image_tuple in training_data:
        label, pixels = image_tuple
        # Stack pixels into single vector
        images[label].append((np.ndarray.flatten(pixels)))
    A = []
    for m in images:
        A.append(np.transpose(np.array(m, dtype="uint8")))
    print("Generated A = [[2**n, m], ...] for digits 0-9")
    print("A[0].shape=",A[0].shape)
    return A

# Put all images into columns
def columnize(image_data):
    columnized_data = []
    for image_tuple in image_data:
        label, pixels = image_tuple
        columnized_tuple = (label, np.ndarray.flatten(pixels))
        columnized_data.append(columnized_tuple)
    return columnized_data

# Takes a columnized image_tuple and
# Ures = [np.identity(784) - u.dot(u.transpose()) for u in Uk]
# Returns a tuple of (guess, actual) integers
def categorize(image_tuple, Ures):
    label, pixels = image_tuple
    bestIndex = -1
    bestResidual = math.inf
    for i in range(0,10):
        residual = sp.norm(Ures[i].dot(pixels))
        if residual < bestResidual: 
            bestIndex = i
            bestResidual = residual
    return bestIndex, label

# print(categorize(testing_data[0]))

# Takes columnized testing_data, a number of tests,
# and Uk = [Uj[:, :k] for Uj in [sp.svd(Aj)[0] for Aj in A]]
# Returns a tuple of (percentage correct,
# dictionary of <which digits confused>: <how often>)
def testCategorize(testing_data, numTests, Uk):
    # Ures is what's needed for calculating residuals
    Ures = [np.identity(784) - u.dot(u.transpose()) for u in Uk]
    numCorrect = 0
    numTests = min(numTests, len(testing_data))
    confused = {}
    for i in testing_data[:numTests]:
        guess, label = categorize(i, Ures)
        if guess == label:
            numCorrect += 1
        else:
            # print(guess, label)
            key = f"{min(guess,label)}-{max(guess,label)}"
            if key in confused.keys():
                confused[f"{min(guess,label)}-{max(guess,label)}"] += 1
            else:
                confused[key] = 1

    # print(f"{numCorrect} correct out of {numTests} tests = {numCorrect / numTests * 100}% accurate.")
    # 9485 correct out of 10000 tests = 94.85% accurate
    # (with k = 10)
    return numCorrect / numTests * 100, confused

# Log results, since they take so long to compute
# for k in range(1, 50, 2)

# Test the above functions

results = []
confused = []

def genResults():

    global results
    results = []
    global confused

    testing_data = list(read("testing"))
    print("Read testing data:", len(testing_data), "images.")

    training_data = list(read())
    print("Read training data:", len(training_data), "images.")
    A = getTrainingA(training_data)

    testing_data = columnize(testing_data)

    print("Generating U for all Aj in A")
    U = [sp.svd(Aj)[0] for Aj in A]

    # Rather than randomly sampling, running on entire data set
    # so as to reproduce results shown in paper
    numTests = 10000

    print("Testing categorization with k= [1...50] and numTests=", numTests)
    for k in range(1,51,1):
        c = testCategorize(testing_data, numTests, [Uj[:, :k] for Uj in U])
        print(f"k={k:>2}: {c[0]:.2f}% accurate.")
        results.append(c[0])
        confused.append(c[1])
        # print("Most confused:", sorted(c[1].items(), key=lambda kv: kv[1])[::-1][:3])

# Previously generated results stored here, to save repeated
# computations. Can be regenerated with genResults()
results = [
    81.84,
    87.16,
    90.26,
    91.54,
    92.63,
    93.44,
    94.02,
    94.21,
    94.77,
    94.85,
    94.93,
    94.89,
    94.98,
    95.09,
    95.27,
    95.24,
    95.39,
    95.56,
    95.63,
    95.73,
    95.73,
    95.68,
    95.85,
    95.78,
    95.76,
    95.79,
    95.68,
    95.82,
    95.82,
    95.74,
    95.72,
    95.65,
    95.78,
    95.74,
    95.83,
    95.74,
    95.75,
    95.72,
    95.64,
    95.70,
    95.61,
    95.63,
    95.55,
    95.44,
    95.41,
    95.23,
    95.09,
    95.01,
    95.03,
    95.14
]

