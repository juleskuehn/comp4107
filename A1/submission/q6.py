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

# Keep this global to avoid recalculating every time
U = -1

# read() function from
# https://gist.github.com/akesling/5358964
# All other code written by Jules
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

def getTrainingA(training_data):
    """
    Format training data into desired form:
    a list A, of matrices Aj with shape (2**n, m)
    for each digit, where j = the digit
    """
    A = [[] for _ in range(10)]
    for image_tuple in training_data:
        label, pixels = image_tuple
        # Stack pixels into single vector
        A[label].append((np.ndarray.flatten(pixels)))
    return [np.array(Aj, dtype="uint8").transpose() for Aj in A]

def columnize(image_data):
    """
    Put all images into columns
    Used on testing data set
    """
    columnized_data = []
    for image_tuple in image_data:
        label, pixels = image_tuple
        # Stack pixels into single vector
        columnized_tuple = (label, np.ndarray.flatten(pixels))
        columnized_data.append(columnized_tuple)
    return columnized_data

def categorize(image_tuple, Ures):
    """
    Takes a columnized image_tuple to compare with
    Ures = [np.identity(784) - u.dot(u.transpose()) for u in Uk]
    Returns a tuple of (guess, actual) integers
    This follows from formula (2) in the paper.
    """
    label, pixels = image_tuple
    bestIndex = -1
    bestResidual = math.inf
    for i in range(0,10):
        residual = sp.norm(Ures[i].dot(pixels))
        if residual < bestResidual: 
            bestIndex = i
            bestResidual = residual
    return bestIndex, label

def testCategorize(testing_data, indices, Uk):
    """
    Takes columnized testing_data, sample indices, and
    Uk = [Uj[:, :k] for Uj in [sp.svd(Aj)[0] for Aj in A]]
    Returns percentage of tests correct
    """
    # Ures is what's needed for calculating residuals
    # Calculate it once to save time in categorize()
    Ures = [np.identity(784) - u.dot(u.transpose()) for u in Uk]
    
    numCorrect = 0
    for i in indices:
        guess, label = categorize(testing_data[i], Ures)
        if guess == label:
            numCorrect += 1
    return numCorrect / len(indices) * 100

def genResults(sample_size):
    """
    Script to replicate the results of paper
    Returns a list of categorization success percentages
    for k = [1...50] for MNIST training and test data
    """
    # Read training and testing data from files,
    # and process into suitable formats
    A = getTrainingA(list(read("testing")))
    testing_data = columnize(list(read("training")))

    global U
    # If not already generated,
    if U == -1:
        # Generate U for all Aj in A
        U = [sp.svd(Aj)[0] for Aj in A]

    if sample_size < len(testing_data):
        sampleIndices = np.random.choice(len(testing_data), sample_size, replace=False)
    else:
        sampleIndices = range(0, len(testing_data))

    results = []
    for k in range(1,51,1):
        c = testCategorize(testing_data, sampleIndices, [Uj[:, :k] for Uj in U])
        results.append(c)

    return results

def plot(results):
    """
    Plot the results as per the paper
    """
    pyplot.plot(results, marker='x', color='r')
    pyplot.xticks(range(0,51,5))
    pyplot.ylabel('Classification Percentage')
    pyplot.xlabel('# of Basis Images')
    pyplot.show()

# Previously generated results stored here, to save repeated
# computations. Can be regenerated with genResults(10000)
# These results are for the entire test set (10000 images).
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

# Plot the results for the entire test set
plot(results)

# Take a random sample of 100 and plot the results
# plot(genResults(100))