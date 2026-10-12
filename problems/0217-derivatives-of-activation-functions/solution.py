import numpy as np
def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	# Your code here
	sig = 1 / (1 + np.exp(-x))
	d_sig = sig * (1 - sig)
	tanh = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
	d_tanh = 1 - tanh ** 2
	d_relu = 0.0 if x <= 0 else 1
	return {'sigmoid': d_sig, 'tanh': d_tanh, 'relu': d_relu}