# COMP 4107, Fall 2018
# Assignment 1, Question 4
#
# Jules Kuehn
# 100661464
#
# Yunkai Wang
# 100968473

import numpy as np
import scipy.linalg as sp

A = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6],
    [1, 1, 1]
])
b = np.array([1, 1, 1, 1]).transpose()

# It is not specified what X should be
# Output will differ based on X's initial value
X = np.array([1, 2, 3]).transpose()

tolerance = 0.01
steps = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]

At = A.transpose()
AtA = np.dot(At, A)
Apinv = np.linalg.pinv(A)


def computeLoss():
    return np.dot(AtA, x) - np.dot(At, b)


def arrToString(arr):
    s = ""
    for el in arr:
        s += f"{el:>12.6f}"
    return s


print("|    Step     |                  x                 |  Iter  |")
print("=============================================================")
print(" P-inv sol'n:", arrToString(np.dot(Apinv, b)))

for step in steps:
    x = X
    L = computeLoss()
    i = 0
    while np.linalg.norm(np.dot(AtA, x) - np.dot(At, b)) > tolerance:
        i += 1
        L = computeLoss()
        x = x - step*L
        if (np.linalg.norm(L) > 100000):
            break
        # print(f" step: {step:.3f}: {arrToString(x)}   {i}")
    print(f" step: {step:.3f}: {arrToString(x)}   {i}")

# print(L)
