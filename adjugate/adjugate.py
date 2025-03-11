import numpy as np


def adjugate_matrix(matrix):
    cofactor = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            det_minor = np.linalg.det(minor)
            cofactor[i, j] = ((-1) ** (i + j)) * det_minor
    return cofactor


def is_equal_adjugate(A):
    if not isinstance(A, np.ndarray):
        return False
    if A.ndim != 2:
        return False
    row, col = A.shape
    if col != row:
        return False
    a_squared = np.dot(A, A)
    a_squared_cofactor = adjugate_matrix(a_squared)
    a_cofactor = adjugate_matrix(A)
    a_cofactor_squared = np.dot(a_cofactor, a_cofactor)

    return np.allclose(a_squared_cofactor, a_cofactor_squared)
