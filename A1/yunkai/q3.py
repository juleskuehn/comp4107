

# import library
import math
import numpy as np
import numpy.linalg as nl
from scipy.spatial import KDTree
from math import sqrt
from numpy import zeros

# populate matrix
delta = 0.001
size = 1401
A = []
for i in range(size):
    row = []
    xi = -0.7 + delta * (i)  # doesn't need to subtract 1 since the index starts from 0
    for j in range(size):
        xj = -0.7 + delta * (j)
        aij = math.sqrt(1 - xi ** 2 - xj ** 2)
        row.append(aij)
    A.append(row)

A = np.array(A)

U, S, V = nl.svd(A, full_matrices=True)
S = np.diag(S)
U2 = U[:, :2]
S2 = S[:2, :2]
V2 = V[:2, :]

# compute rank-2 matrix A2
A2 = U2.dot(S2.dot(V2))
print('A2=', A2)

diff = np.linalg.norm(A-A2)
print("\n||A - A2|| =", diff)
