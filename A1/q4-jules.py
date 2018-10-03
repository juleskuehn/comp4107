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

# x is a Vector
# http://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf
# and last slides on
# https://sikaman.dyndns.org:8443/WebSite/rest/site/courses/4107/handouts/04-BasicAlgorithms.pdf
#
# If xHat == 0 , xHat solves Ax-b (rarely happens)
# Least squares solution minimizes ||Ax-b||^2
# Some values of step size will not have solutions
# because it will overshoot solutions
tolerance = 0.01
steps = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.05]
b = np.array([1,1,1,1]).transpose()
At = A.transpose()
AtA = np.dot(At, A)

# print(b)
# print(At)
# print(AtA)

Apinv = np.linalg.pinv(A)
X = np.array([0,0,0]).transpose()
def computeLoss():
    return np.dot(AtA, x) - np.dot(At, b)

x = X
L = computeLoss()
i = 0
for step in steps:
    x = X
    L = computeLoss()
    i = 0
    print(f"step = {step}:")
    while np.linalg.norm(L) > tolerance:
        i += 1
        L = computeLoss()
        x = x - step*L
        # if i%100 == 0:
        #     print(i, x)
    print(i, x)

print("Pseudo-inverse solution = ", np.dot(Apinv, b))
# print(L)