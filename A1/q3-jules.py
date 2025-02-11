# COMP 4107, Fall 2018
# Assignment 1, Question 3
#
# Jules Kuehn
# 100661464
#
# Yunkai Wang
# 100968473

import numpy as np
import scipy.linalg as sp
import math

def f(i, j):
    x = -0.7 + 0.001*(i)
    y = -0.7 + 0.001*(j)
    return math.sqrt(1 - x**2 - y**2)

g = np.vectorize(f)
A = np.fromfunction(g, (1401, 1401))

U, s, V = sp.svd(A, full_matrices=True)

# Diagonalize s vector to matrix
sDiag = np.diag(s)

print("\nA =\n", A)

# The best rank(2) matrix takes the first 2 values from s
# since s is sorted descending
U2 = U[:, :2]
V2 = V[:2, :]
s2 = sDiag[:2, :2]

A2 = np.dot(U2, s2).dot(V2)

print("\nA2 =\n", A2)

diff = np.linalg.norm(A-A2)
print("\n||A - A2|| =", diff)

# Accuracy of this approximation is very good
print(f"\n||A - A2|| / ||A|| = {diff*100 / np.linalg.norm(A):5.3f}%")
