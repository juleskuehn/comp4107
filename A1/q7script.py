
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
    count = 0
    data = base + test

    for rate in data:
        info = rate.split()
        userID = int(info[0]) - 1  # index from 0 to 942
        itemID = int(info[1]) - 1
        rating = int(info[2])
        if count <= training_rate * n:
            R[userID][itemID] = rating
        else:
            testData.append((userID, itemID, rating))

        count += 1
    return R, testData

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

# result = [
#     1.13694666392258,
#     1.1398356632016955,
#     1.1411625804179784,
#     1.1451504272170445,
#     1.147375865206018,
#     1.1508316314076712,
#     1.1524357093926474
# ]

result08 = [
    1.1094629684891646,
    1.1083705958779786,
    1.1084692864635652,
    1.1086785809309723,
    1.1086618853869186,
    1.1076878951743228,
    1.1081233148382676
]
result05 = [
    2.520464846246617,
    2.5205009990523894,
    2.5205009990523894,
    2.520500999052389,
    2.520500999052389,
    2.5205009990523903,
    2.520500999052389
]
result02 = [
    3.016864079459395,
    3.0168640794593955,
    3.0168640794593955,
    3.0168640794593955,
    3.0168640794593955,
    3.0168640794593955,
    3.016864079459395
]

training_rates = [0.8, 0.5, 0.2]

'''
result08 = []
result05 = []
result02 = []
for training_rate in training_rates:
    result = []
    R, testData = read_data(training_rate)

    print("Using ", training_rate, " as training/test rate")
    for basis_size in range(600, 901, 50):
        error = 0.0
        count = 0

        # just used to save calculation
        prevAvgRating = 0.0
        prevUserID = -1

        UKM, SKSQRT, VK = decompose_matrix(R, basis_size)

        for rate in testData:
            userID = int(rate[0]) - 1  # index from 0 to 942
            itemID = int(rate[1]) - 1
            rating = int(rate[2])

            if prevUserID == userID:
                userAverageRating = prevAvgRating
            else:

                # calculate average of row i
                numRating = 0
                totalRating = 0.0
                for userRating in R[userID]:
                    if userRating != 0:
                        # print(userRating)
                        numRating += 1
                        totalRating += userRating

                if numRating != 0:
                    userAverageRating = totalRating / numRating
                else:
                    userAverageRating = 0
                prevUserID = userID
                prevAvgRating = userAverageRating

            firstHalf = UKM.dot(SKSQRT.transpose())[userID]
            secondHalf = SKSQRT.dot(VK)[:, itemID]
            product = firstHalf.dot(secondHalf)
            predict = userAverageRating + product
            error += abs(predict - rating)
            count += 1

        print("Basis size is ", basis_size, "Error is ", error / count)

        if training_rate == 0.8:
            result08.append(error / count)
        elif training_rate == 0.5:
            result05.append(error / count)
        else:
            result02.append(error / count)
    print("---------------------------------")
'''

pyplot.plot([size for size in range(600, 901, 50)], result08, marker='x', color='r')
pyplot.plot([size for size in range(600, 901, 50)], result05, marker='x', color='g')
pyplot.plot([size for size in range(600, 901, 50)], result02, marker='x', color='b')

pyplot.xticks(range(600, 900, 50))
pyplot.ylabel('MAE')
pyplot.xlabel('Folding-in Model Size')
pyplot.show()
