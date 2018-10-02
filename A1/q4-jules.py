# COMP 4107, Fall 2018
# Assignment 1, Question 4
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

# x is a Vector
# http://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf
# and last slides on
# https://sikaman.dyndns.org:8443/WebSite/rest/site/courses/4107/handouts/04-BasicAlgorithms.pdf
#
# xHat = solution = PseudoInverse(A) * b
# If xHat == 0 , xHat solves Ax-b (rarely happens)
# Least squares solution minimizes ||Ax-b||^2
# Some values of step size will not have solutions
# because it will overshoot solutions
tolerance = 0.01
steps = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
b = np.array([1,1,1,1]).transpose()
