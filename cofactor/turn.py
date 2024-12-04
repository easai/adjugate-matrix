import numpy as np

# Define the matrix A
A = np.array([[7, -5, -8],
              [-3, -7, -2],
              [0, -4, -8]])

# Calculate the square of the matrix A
A_squared = np.dot(A, A)

# Function to calculate the cofactor matrix
def cofactor_matrix(matrix):
    cofactor = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            det_minor = np.linalg.det(minor)
            cofactor[i, j] = ((-1) ** (i + j)) * det_minor
    return cofactor

# Calculate the cofactor matrix
A_cofactor = cofactor_matrix(A_squared)
B = cofactor_matrix(A)
A_cofactor_squared = np.dot(B,B)

# Print the results
print("Square of matrix A:\n", A_cofactor)
print("\nCofactor matrix of A:\n", A_cofactor_squared)

# Verify if the square of the matrix equals the cofactor matrix
print("\nAre the square of the matrix and cofactor matrix equal?")
print(np.allclose(A_squared, A_cofactor))