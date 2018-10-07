# COMP 4107, Fall 2018
# Assignment 1, Question 5
#
# Jules Kuehn
# 100661464
#
# Yunkai Wang
# 100968473

import numpy as np
import scipy.linalg as sp
import sympy

A = np.array([
    [3, 2 , -1, 4],
    [1, 0, 2, 3],
    [-2, -2, 3, -1]
])

# Let's try putting the matrix in reduced row echelon form,
# to check if the rows are independent
# See https://docs.sympy.org/dev/tutorial/matrices.html#rref

rref, indRows = sympy.Matrix(A).rref()
print("\nA in rref:\n", rref)
print("Independent rows:", len(indRows))

# To check if the columns are independent, transpose and then
# put into reduced row echelon form
rrefT, indCols = sympy.Matrix(A).T.rref()
print("\nA^T in rref:\n", rrefT)
print("Independent cols:", len(indCols))