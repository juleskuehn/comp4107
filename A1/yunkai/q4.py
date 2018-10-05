
import numpy as np

A = np.array([
    [1, 2, 3],
    [2, 3, 4],
    [4, 5, 6],
    [1, 1, 1]
])

b = np.array([1, 1, 1, 1]).transpose()
X = np.array([1, 2, 3]).transpose()

# learning rate
lr = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5]

tolerance = 0.01

AT = A.transpose()
ATA = np.dot(AT, A)
ATB = np.dot(AT, b)
ATBSQ = ATB ** 2
Apinv = np.linalg.pinv(A)


def arrToString(arr):
    s = ""
    for el in arr:
        s += f"{el:>12.6f} "
    return s

print(" P-inv sol'n:", arrToString(np.dot(Apinv, b)))
print("|    Step     | RESULT |                  x                 |  Iter  |")
print("=============================================================")

def computeLoss(x):
    return np.dot(ATA, x) - np.dot(AT, b)

for step in lr:
    success = True
    x = X
    L = computeLoss(x)
    i = 0
    while np.linalg.norm(np.dot(ATA, x)**2 - np.dot(AT, b)**2) > tolerance:
        i += 1
        L = computeLoss(x)
        x = x - step*L

        # diff is too big, impossible to get the correct result
        if (np.linalg.norm(L) > 100000):
            success = False
            break

    result = "failed " if not success else "success"
    print(f" step: {step:.3f}:  {result} {arrToString(x)}   {i}")
