# COMP 4107, Fall 2018
# Assignment 1, Question 3
#
# Jules Kuehn
# 100661464

import numpy as np
import scipy.linalg as sp
import math

def f(i, j):
    x = -0.7 + 0.001*(i - 1)
    y = -0.7 + 0.001*(j - 1)
    return math.sqrt(1 - x**2 - y**2)

g = np.vectorize(f)
A = np.fromfunction(g, (1401, 1401))

U, s, V = sp.svd(A, full_matrices=True)

# Diagonalize s vector to matrix
s = np.diag(s)

print("\nA =\n", A)
print("\nU =\n", U)
print("\nS =\n", s)
print("\nV =\n", V)