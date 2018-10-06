
# import library
import numpy as np
import numpy.linalg as nl
from math import sqrt
from numpy import zeros
from matplotlib import pyplot

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

    U, S, V = nl.svd(basis, full_matrices=True)
    S = np.diag(S)

    # reduce the space of these vectors
    UK = U[:, :k]
    SK = S[:k, :k]
    VK = V[:k, :]

    projection = foldInUser.dot(VK.transpose()).dot(nl.inv(SK))

    # U(k+m) with all users' ratings
    UKM = np.concatenate((UK, projection))
    SKSQRT = np.sqrt(SK)
    return UKM, SKSQRT, VK

training_rates = [0.8, 0.5, 0.2]
result08 = []
result05 = []
result02 = []
for training_rate in training_rates:
    result = []
    R, testData = read_data(training_rate)
    basis_size = 943

    print("Using ", training_rate, " as training/test rate")
    for basis_size in range(600, 601, 50):
        error = 0.0
        count = len(testData)
        # just used to save calculation
        # prevAvgRating = 0.0
        # prevUserID = -1

        UKM, SKSQRT, VK = decompose_matrix(R, basis_size)

        for rating in testData:
            userID = rating[0] # index from 0 to 942
            itemID = rating[1]
            userRating = rating[2]

            userRated = np.count_nonzero(R[userID])
            # Should the initial average be 2.5, not 0, if the user hasn't rated anything yet?
            userAverageRating = 2.5 if userRated == 0 else sum(R[userID]) / userRated

            firstHalf = UKM.dot(SKSQRT.transpose())[userID]
            secondHalf = SKSQRT.dot(VK)[:, itemID]
            product = firstHalf.dot(secondHalf)
            predictRating = userAverageRating + product
            error += abs(predictRating - userRating)

        print("Basis size is ", basis_size, "Error is ", error, "/", count, "=", error/count)
        
        if training_rate == 0.8:
            result08.append(error / count)
        elif training_rate == 0.5:
            result05.append(error / count)
        else:
            result02.append(error / count)
    print("---------------------------------")


# pyplot.plot([size for size in range(600, 901, 50)], result08, marker='x', color='r')
# pyplot.plot([size for size in range(600, 901, 50)], result05, marker='x', color='g')
# pyplot.plot([size for size in range(600, 901, 50)], result02, marker='x', color='b')

# pyplot.xticks(range(600, 901, 50))
# pyplot.ylabel('MAE')
# pyplot.xlabel('Folding-in Model Size')
# pyplot.show()
