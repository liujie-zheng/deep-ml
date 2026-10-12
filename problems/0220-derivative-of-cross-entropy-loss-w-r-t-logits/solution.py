import numpy as np
def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	# Your code here
	x = np.array(logits)
	e_x = np.exp(x - np.max(x))
	softmax_x = e_x / np.sum(e_x)

	gradient = softmax_x.copy()
	gradient[target] -= 1
	return gradient