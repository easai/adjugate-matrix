import numpy as np
from cofactor.adjugate import is_equal_adjugate


def test_matrix():
    A = np.array([[7, -5, -8],
                  [-3, -7, -2],
                  [0, -4, -8]])
    assert is_equal_adjugate(A)


def test_uppder_triangular():
    A = np.array([[7, -5, -8],
                  [0, -7, -2],
                  [0, 0, -8]])
    assert is_equal_adjugate(A)


def test_lower_triangular():
    A = np.array([[7, 0, 0],
                  [-3, -7, 0],
                  [0, -4, -8]])
    assert is_equal_adjugate(A)


def test_array():
    A = np.array([1])
    assert not is_equal_adjugate(A)


def test_empty():
    A = np.array([])
    assert not is_equal_adjugate(A)


def test_non_square():
    A = np.array([[7, -5, -8], [0, 0, -8]])
    assert not is_equal_adjugate(A)
