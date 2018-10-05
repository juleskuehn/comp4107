
import numpy as np

A = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6],
    [1, 1, 1]
])

U, s, V = np.linalg.svd(A, full_matrices=True)

# Diagonalize s vector to matrix
s = np.diag(s)

print("A=", A)
print("U=", U)
print("S=", s)
print("V=", V)
