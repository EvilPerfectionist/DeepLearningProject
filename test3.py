from sklearn.neighbors import NearestNeighbors
import numpy as np
Y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Z = Y[X]
print(Z)
