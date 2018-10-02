# COMP 4107, Fall 2018
# Assignment 1, Question 2
#
# Jules Kuehn
# 100661464

import numpy as np
import scipy.linalg as sp

A = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6],
    [1, 1, 1]
])

U, s, V = sp.svd(A, full_matrices=True)

# Diagonalize s vector to matrix
s = np.diag(s)

print("\nA =\n", A)
print("\nU =\n", U)
print("\nS =\n", s)
print("\nV =\n", V)