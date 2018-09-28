# COMP 4107, Fall 2018
# Assignment 1, Question 1
#
# Jules Kuehn
# 100661464

import numpy as np
import scipy.linalg as sp

A = np.array([
    [3, 1, 2, 3],
    [4, 3, 4, 3],
    [3, 2, 1, 5],
    [1, 6, 5, 2]
])

U, s, V = sp.svd(A, full_matrices=True)

print("\nA =\n", A)
print("\nU =\n", U)
print("\nS =\n", s)
print("\nV =\n", V)

smat = np.diag(s)
U2 = U[:, :2]
s2i = np.linalg.inv(smat[:2, :2])

print("\nsmat =\n", smat)
print("\nU2 =\n", U2)
print("\ns2i =\n", s2i)

Alice = np.array([5, 3, 4, 4])
Alice2D = np.dot(Alice, U2).dot(s2i)

print("\nAlice2D =\n", Alice2D)