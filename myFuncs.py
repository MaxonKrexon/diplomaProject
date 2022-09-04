import numpy as np


def cross(A=None, B=None):
    x = 0
    y = 1
    z = 2

    result = np.array([A[z] * B[y] - A[y] * B[z], A[x] * B[z] - A[z] * B[x], A[y] * B[x] - A[x] * B[y]], np.float64)
    return result

def strekeAB(A=None, B=None):
    if sum(B) == 0:
        y = np.sqrt(sum(A ** 2))
    else:
        y = np.sqrt(sum((A - B) ** 2))
    return y

def dot(A = None, B = None):
    if np.size(A, 0) == np.size(B, 0):
        C = 0.0
        for j in range(np.size(A, 0)):
            C += np.float64(A[j]*B[j])
        return C
    else:
        print("Arrays must be same size")
        return


