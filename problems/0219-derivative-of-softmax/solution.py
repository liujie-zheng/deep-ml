import numpy as np
def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	# Your code here
	x = np.array(x)
	exp_x = np.exp(x)
	softmax_x = exp_x / np.sum(exp_x)
	n = x.shape[0]
	matrix = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			if i == j:
				matrix[i, j] = softmax_x[i] * (1 - softmax_x[i])
			else:
				matrix[i, j] = -softmax_x[i] * softmax_x[j]
	return matrix
