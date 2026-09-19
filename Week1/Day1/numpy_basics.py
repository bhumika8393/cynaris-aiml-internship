import numpy as np

# -------------------------------
# 1. Create a 1D Array
# -------------------------------
array_1d = np.array([10, 20, 30, 40, 50])

print("1D Array:")
print(array_1d)
print("Shape:", array_1d.shape)

# -------------------------------
# 2. Create a 2D Array
# -------------------------------
array_2d = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("\n2D Array:")
print(array_2d)
print("Shape:", array_2d.shape)

# -------------------------------
# 3. Create a 3D Array
# -------------------------------
array_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("\n3D Array:")
print(array_3d)
print("Shape:", array_3d.shape)

# -------------------------------
# 4. Broadcasting
# -------------------------------
print("\nBroadcasting Example:")
print(array_1d + 5)

# -------------------------------
# 5. Vectorized Operations
# -------------------------------
print("\nVectorized Operations:")

numbers = np.array([2, 4, 6, 8])

print("Original:", numbers)
print("Multiply by 2:", numbers * 2)
print("Square:", numbers ** 2)
print("Divide by 2:", numbers / 2)

# -------------------------------
# 6. Matrix Multiplication
# -------------------------------
matrix1 = np.array([
    [1, 2],
    [3, 4]
])

matrix2 = np.array([
    [5, 6],
    [7, 8]
])

result = np.matmul(matrix1, matrix2)

print("\nMatrix Multiplication:")
print(result)