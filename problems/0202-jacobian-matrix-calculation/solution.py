import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	# Your code here
	f_x = np.array(f(x)) # (m, )
	m = len(f_x)
	n = len(x)

	matrix = np.zeros((m, n))
	
	for j in range(n):
		xp = x.copy()
		xp[j] += h # (n, )
		f_xp = np.array(f(xp))
		diff = (f_xp - f_x) / h # (m, )
		matrix[:, j] = diff # (m, 1)

	return matrix
