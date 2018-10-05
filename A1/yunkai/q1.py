

# import library
import numpy as np
import numpy.linalg as nl
from scipy.spatial import KDTree

'''
matrix of ratings by users for items
copied from lecture note 'linear algebra'
page 49
'''
X = np.array([
    [3, 1, 2, 3],
    [4, 3, 4, 3],
    [3, 2, 1, 5],
    [1, 6, 5, 2]])

'''
Alice's rating for items, also copied from
page 49
'''
Alice = np.array([5, 3, 4, 4])

U, S, V = nl.svd(X, full_matrices=True)

# Diagonolize vector into matrix
S = np.diag(S)

# Reduce to 2D
U2 = U[:, :2]
S2 = S[:2, :2]
V2 = V[:2, :]

Alice2D = np.dot(Alice, U2).dot(nl.inv(S2))
print('Alice2D=', Alice2D)

Alice4D = np.dot(Alice, U).dot(nl.inv(S))
print('Alice4D=', Alice4D)

Predict = np.average(Alice) + np.dot(V2[:, 0], S2).dot(U2[3])
print('Prediction for Alice on EPL is ', Predict)

tree2D = KDTree(V2.transpose())
closestAlice2D = tree2D.query(Alice2D)[1]
tree4D = KDTree(V.transpose())
closestAlice4D = tree4D.query(Alice4D)[1]

print('Closest user in 2D is ', closestAlice2D)
print('Closest user in 4D is ', closestAlice4D)
