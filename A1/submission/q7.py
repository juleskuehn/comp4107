# COMP 4107, Fall 2018
# Assignment 1, Question 7
#
# Jules Kuehn
# 100661464
#
# Yunkai Wang
# 100968473

import time
import numpy as np
import numpy.linalg as nl
from math import sqrt
from numpy import zeros
from matplotlib import pyplot
import scipy.linalg as sp

# the optimal value of k
k = 14

# the total number of ratings
n = 100000

def read_data(training_rate=0.8):
    base = [line.rstrip('\n') for line in open('./ml-100k/u1.base', 'r')]
    test = [line.rstrip('\n') for line in open('./ml-100k/u1.test', 'r')]

    # user-movie matrix R
    # 943 users' ratings on 1682 items
    R = zeros((943, 1682))

    testData = []
    i = 0

    # All 100,000 ratings
    data = base + test

    for rating in data:
        info = rating.split()
        userID = int(info[0]) - 1  # index from 0 to 942
        itemID = int(info[1]) - 1  # index from 0 to 1681
        rating = int(info[2])
        if i < int(training_rate * n):
            R[userID][itemID] = rating
        else:
            testData.append((userID, itemID, rating))
        i += 1

    return R, testData

# Returns U, sqrt(S), and V transpose
def decompose_matrix(R, n=600):
    basis = R[:n]
    foldInUser = R[n:]

    U, s, Vt = nl.svd(basis, full_matrices=True)
    S = np.diag(s)

    # reduce the space of these vectors
    Uk = U[:, :k]
    Sk = S[:k, :k]
    Vkt = Vt[:k, :]

    projection = foldInUser.dot(Vkt.transpose()).dot(nl.inv(Sk))

    # U(k+m) with all users' ratings
    Ukm = np.concatenate((Uk, projection))
    return Ukm, Sk, Vkt

def fillAndNormalize(R):
    rows, cols = R.shape

    # FILL:
    # Get overall average rating for all films
    numRatings = np.count_nonzero(R)
    avgRating = np.sum(R) / numRatings
    for j in range(0, cols):
        # Get average rating for specific film
        itemCol = R[:,j]
        numRatings = np.count_nonzero(itemCol)
        # If the film has never been rated, give it the overall average
        itemAvgRating = (avgRating if numRatings == 0
                else sum(itemCol) / numRatings)
        # Fill in empty ratings with item average
        for i in range(0, rows):
            if R[i, j] == 0:
                R[i, j] = itemAvgRating

    # NORMALIZE:
    # Need to store user averages before normalization
    userAverages = []
    # Subtract user average from each rating
    for i in range(0, rows):
        userAvgRating = np.average(R[i])
        userAverages.append(userAvgRating)
        R[i] = R[i] - userAvgRating

    return R, userAverages

training_rates = [0.8, 0.5, 0.2]
result08 = []
performance08 = []
result05 = []
performance05 = []
result02 = []
performance02 = []
for training_rate in training_rates:
    R, testData = read_data(training_rate)
    R, userAverages = fillAndNormalize(R)

    print("Using ", training_rate, " as training/test rate")
    for basis_size in range(600, 901, 50):
        error = 0.0
        count = len(testData)

        Uk, Sk, Vkt = decompose_matrix(R, basis_size)

        ts = time.time()

        for rating in testData:
            userID = rating[0] # index from 0 to 942
            itemID = rating[1]
            actual = rating[2]

            pseudoUsers = Uk.dot(np.sqrt(Sk))
            pseudoFilms = np.sqrt(Sk).dot(Vkt)

            Pij = (userAverages[userID]
                    + pseudoUsers[userID].dot(pseudoFilms[:,itemID]))
            error += abs(Pij - actual)

        caltime = time.time() - ts

        print("Basis size:", basis_size, ", Error:", error/count,
              ", Number of test data: ", count, ", calculation time:", caltime,
              ", Throughput:", count/caltime)

        if training_rate == 0.8:
            result08.append(error / count)
            performance08.append(count / caltime)
        elif training_rate == 0.5:
            result05.append(error / count)
            performance05.append(count / caltime)
        else:
            result02.append(error / count)
            performance02.append(count / caltime)
    print("---------------------------------")

pyplot.figure(1)
pyplot.plot([size for size in range(600, 901, 50)], result08, marker='x', color='r')
pyplot.plot([size for size in range(600, 901, 50)], result05, marker='x', color='g')
pyplot.plot([size for size in range(600, 901, 50)], result02, marker='x', color='b')

pyplot.xticks(range(600, 901, 50))
pyplot.ylabel('MAE')
pyplot.xlabel('Folding-in Model Size')
pyplot.show()

pyplot.figure(2)
pyplot.plot([size for size in range(600, 901, 50)], performance08, marker='x', color='r')
pyplot.plot([size for size in range(600, 901, 50)], performance05, marker='x', color='g')
pyplot.plot([size for size in range(600, 901, 50)], performance02, marker='x', color='b')

pyplot.xticks(range(600, 901, 50))
pyplot.ylabel('Throughput')
pyplot.xlabel('Basis Size')
pyplot.show()
