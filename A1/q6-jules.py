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

training_data = list(read())
print(len(training_data))
label, pixels = training_data[0]
images = [
    [], [], [], [], [], [], [], [], [], []
]
for image_tuple in training_data:
    label, pixels = image_tuple
    # Stack pixels into single vector
    i = len(images[label])
    images[label].append([])
    # print(images[label])
    for row in pixels:
        for pixel in row:
            images[label][i].append(pixel)

A = []
Asvd = []
U = []
Aavg = []

for m in images:
    A.append(np.transpose(np.array(m, dtype="uint8")))
    # Asvd.append(sp.svd(A[-1], full_matrices=True))
    # tmpU, s, V = Asvd[-1]
    # U.append(tmpU)
    Aavg.append(np.array(np.sum(A[-1], axis=1) / A[-1].shape[0]))

print("Generated A = [[2**n, m], ...] for digits 0-9")
print("A[0].shape=",A[0].shape)

def reconstruct(Aa):
    tmp = []
    for i in range(0, 28):
        tmp.append([])
        for j in range(0, 28):
            tmp[i].append(Aa[i*28 + j])
    return np.array(tmp, dtype=np.uint8)

def showSum(A, i):
    show(reconstruct(np.array(np.sum(A[i], axis=1) / A[i].shape[0])))

def kRank(A, k):
    U, s, V = sp.svd(A, full_matrices=True)

    # Diagonalize s vector to matrix
    sDiag = np.diag(s)

    # print("\nA =\n", A)

    # The best rank(2) matrix takes the first 2 values from s
    # since s is sorted descending
    Uk = U[:, :k]
    Vk = V[:k, :]
    sk = sDiag[:k, :k]

    A2 = np.dot(Uk, sk).dot(Vk)

    return A2

def kRankU(A, k):
    U, s, V = sp.svd(A, full_matrices=True)
    return U[:, :k]

def bestRank(A, proportionEnergy):
    # What rank of matrix contains >= 99.9% energy?
    U, s, V = sp.svd(A, full_matrices=True)
    sumS = np.sum(s)
    k = 0
    e = 0
    while e < sumS * proportionEnergy :
        e += s[k]
        k += 1

    # print("\nbest k =", k)

    Ak = np.dot(U[:, :k], (np.diag(s))[:k, :k]).dot(V[:k, :])

    # print(f"\nA{k} =\n", Ak)

    # diff = np.linalg.norm(A-Ak)
    # print(f"\n||A - A{k}|| =\n", diff)
    # print(f"\n||A - A{k}|| / ||A|| = {diff / np.linalg.norm(A):10.10}") 

    return Ak, k

# Generate the Ak approx
U10 = [kRankU(a, 10) for a in A]


testing_data = list(read("testing"))
print(len(testing_data))
label, pixels = testing_data[0]



def categorize(image_tuple):
    label, pixels = image_tuple
    # Stack pixels into single vector
    col = []
    # print(images[label])
    for row in pixels:
        for pixel in row:
            col.append(pixel)
    # Now we have column vector ready to test
    bestIndex = 0
    bestResidual = math.inf
    for i in range(0,10):
        residual = sp.norm(
            (np.identity(784) - U10[i].dot(U10[i].transpose())).dot(col)
        )
        if residual < bestResidual: 
            bestIndex = i
            bestResidual = residual
    return bestIndex, label

print(categorize(testing_data[0]))

numCorrect = 0
for i in testing_data:
    guess, label = categorize(i)
    if guess == label:
        numCorrect += 1
    else:
        print(guess, label)

print(f"{numCorrect} correct out of {len(testing_data)} tests = {numCorrect / len(testing_data) * 100}% accurate.")
