from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	# Your code here
	n = len(point)
	H = np.zeros((n, n))
	x = np.array(point)
	for i in range(n):
		for j in range(n):
			if i == j:
				ei = np.zeros(n)
				ei[i] = 1
				hei = h * ei
				H[i, i] = (f(x + hei) - 2 * f(x) + f(x - hei)) / (h ** 2)
			else:
				ei = np.zeros(n)
				ei[i] = 1
				hei = h * ei
				ej = np.zeros(n)
				ej[j] = 1
				hej = h * ej
				H[i, j] = (f(x + hei + hej) - f(x + hei - hej) - f(x - hei + hej) + f(x - hei - hej)) / (4 * h ** 2)
	return H