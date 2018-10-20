# COMP 4107, Fall 2018
# Assignment 1, Question 1
#
# Jules Kuehn
# 100661464
#
# Yunkai Wang
# 100968473

import numpy as np
import scipy.linalg as sp
from scipy.spatial import KDTree

'''
matrix of A by users for items
copied from lecture note 'linear algebra'
page 49
'''
A = np.array([
    [3, 1, 2, 3],
    [4, 3, 4, 3],
    [3, 2, 1, 5],
    [1, 6, 5, 2]
])

'''
Alice's rating for items, also copied from
page 49
'''
Alice = np.array([5, 3, 4, 4])

U, s, V = sp.svd(A, full_matrices=True)

# U and V are opposite compared to slides
# and some signs are reversed, leading me
# to believe that slides have errors
print("\nA =\n", A)
print("\nU =\n", U)
print("\nS =\n", s)
print("\nV =\n", V)

# Diagonalize s vector to matrix
s = np.diag(s)

# Reduce U, s, V to 2 dimensions
U2 = U[:, :2]
s2 = s[:2, :2]
V2 = V[:2, :]

alicia = A[0]
bob    = A[1]
mary   = A[2]
sue    = A[3]
# Alice is not included in A, but we can generate
# a corresponding representation.
#
# I expect that the U and V are switched (at times)
# in slides because this actually works.
Alice2D = np.dot(Alice, U2).dot(np.linalg.inv(s2))
Alicia2D = np.dot(alicia, U2).dot(np.linalg.inv(s2))
Bob2D = np.dot(bob, U2).dot(np.linalg.inv(s2))
Mary2D = np.dot(mary, U2).dot(np.linalg.inv(s2))
Sue2D = np.dot(sue, U2).dot(np.linalg.inv(s2))

print("\nAlice2D =\n", Alice2D)

# Predict Alicia's EPL
PredictAliciaEPL = np.average(A[0]) + U2[0,:].dot(s2).dot(V2[:,3])
print("\nPredictAliceEPL =\n", PredictAliciaEPL)

# Finding the closest user to Alice in 2D and 4D
# Note that I am using V rather than U and transposing
# due to the discrepancy with the slides.
#
# KDtree idea taken from https://stackoverflow.com/questions/32446703/find-closest-vector-from-a-list-of-vectors-python
# This could also be accomplished by looping and finding norms,
# then the minimum of the norms.
tree2D = KDTree(U2)
closestAlice2D = tree2D.query(Alice2D)[1]
tree4D = KDTree(U)
closestAlice4D = tree4D.query(Alice)[1]

print("\nClosest to Alice in 2D and 4D =\n", closestAlice2D, closestAlice4D)

# So the closest in 2D is Alicia, and the closest in 4D is Mary