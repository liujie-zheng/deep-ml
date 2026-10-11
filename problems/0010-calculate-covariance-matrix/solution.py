import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	vectors = np.array(vectors) # (2, 3)
	means = np.mean(vectors, axis=1, keepdims=True) # (2, 1)
	deviations = vectors - means # (2, 3)
	sums = deviations @ deviations.T
	cov = sums / (vectors.shape[1] - 1)
	return cov.tolist()