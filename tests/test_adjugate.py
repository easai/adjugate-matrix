import numpy as np
from adjugate.adjugate import is_equal_adjugate

def test_matrix():
    """
    Test if the adjugate of the square of a given square matrix A
    is equal to the square of the adjugate of matrix A.
    """
    A = np.array([[7, -5, -8],
                  [-3, -7, -2],
                  [0, -4, -8]])
    assert is_equal_adjugate(A)

def test_upper_triangular():
    """
    Test if the adjugate of the square of a given upper triangular matrix A
    is equal to the square of the adjugate of matrix A.
    """
    A = np.array([[7, -5, -8],
                  [0, -7, -2],
                  [0, 0, -8]])
    assert is_equal_adjugate(A)

def test_lower_triangular():
    """
    Test if the adjugate of the square of a given lower triangular matrix A
    is equal to the square of the adjugate of matrix A.
    """
    A = np.array([[7, 0, 0],
                  [-3, -7, 0],
                  [0, -4, -8]])
    assert is_equal_adjugate(A)

def test_array():
    """
    Test if is_equal_adjugate function correctly returns False
    for a one-dimensional array A.
    """
    A = np.array([1])
    assert not is_equal_adjugate(A)

def test_empty():
    """
    Test if is_equal_adjugate function correctly returns False
    for an empty array A.
    """
    A = np.array([])
    assert not is_equal_adjugate(A)

def test_non_square():
    """
    Test if is_equal_adjugate function correctly returns False
    for a non-square matrix A.
    """
    A = np.array([[7, -5, -8], [0, 0, -8]])
