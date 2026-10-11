import numpy as np

def calculate_correlation_matrix(X, Y=None):
	# Your code here
	if Y is None:
		Y = X
	
	dev_x = X - np.mean(X, axis=0) # (3, 2)
	dev_y = Y - np.mean(Y, axis=0)

	cov = (1 / X.shape[0]) * (dev_x.T @ dev_y) #(2, 2)

	std_x = np.std(X, axis=0, keepdims=True) # (1, 2)
	std_y = np.std(Y, axis=0, keepdims=True)

	return cov / (std_x.T @ std_y)