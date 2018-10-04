
# import library
import numpy as np
import numpy.linalg as nl
from math import sqrt
from numpy import zeros

# the optimal value of k
k = 14

# optimal value of the basis size
n = 600

# user-movie matrix R
# 943 users' ratings on 1682 items
R = zeros((943, 1682))

# read the user rating data
# read u1.base which contains 80% of the data
# leave the rest 20% as test data
data = [line.rstrip('\n') for line in open('./ml-100k/u1.base', 'r')]
test = [line.rstrip('\n') for line in open('./ml-100k/u1.test', 'r')]
count = 0
for rate in data:
    info = rate.split()
    userID = int(info[0]) - 1  # index from 0 to 942
    itemID = int(info[1]) - 1
    rating = int(info[2])

    R[userID][itemID] = rating

# take the first 600 users' rates as basis
basis = R[:n]

# all the users that need to be fold in to the matrix
foldInUser = R[n:]

U, S, V = nl.svd(basis, full_matrices=True)
S = np.diag(S)

# reduce the space of these vectors
UK = U[:, :k]
SK = S[:k, :k]
VK = V[:k, :]

# reduced user-item matrix AK
AK = UK.dot(SK).dot(VK)

projection = foldInUser.dot(VK.transpose()).dot(nl.inv(SK))

# U(k+m) with all users' ratings
UKM = np.concatenate((UK, projection))

SKSQRT = np.sqrt(SK)

error = 0.0
count = 0
prevAvgRating = 0.0
prevUserID = -1
for rate in test:
    info = rate.split()
    userID = int(info[0]) - 1  # index from 0 to 942
    itemID = int(info[1]) - 1
    rating = int(info[2])

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

        userAverageRating = totalRating / numRating
        prevUserID = userID
        prevAvgRating = userAverageRating

    firstHalf = UK.dot(SKSQRT.transpose())[userID]
    secondHalf = SKSQRT.dot(VK)[:, itemID]
    product = firstHalf.dot(secondHalf)
    predict = userAverageRating + product
    # if product < 0:
    #     print('met once ', count)
    #     break
    error += abs(predict - rating)
    count += 1

print("Error is ", error / count)
