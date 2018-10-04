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
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
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

for m in images:
    A.append(np.array(m))

print(label)
print(pixels)
# show(pixels)

# Stack 