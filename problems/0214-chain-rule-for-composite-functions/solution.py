import numpy as np

def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
	"""
	Compute derivative of composite functions using chain rule.
	
	Args:
		functions: List of function names (applied right to left)
		          Available: 'square', 'sin', 'exp', 'log'
		x: Point at which to evaluate derivative
	
	Returns:
		Derivative value at x
	
	Example:
		['sin', 'square'] represents sin(x²)
		['exp', 'sin', 'square'] represents exp(sin(x²))
	"""
	# Your code here
	prod = 1
	u = x
	for func in functions[::-1]:
		if func == 'square':
			prod *= 2 * u
			u *= u
		elif func == 'sin':
			prod *= np.cos(u)
			u = np.sin(u)
		elif func == 'log':
			prod /= u
			u = np.log(u)
		elif func == 'exp':
			prod *= np.exp(u)
			u = np.exp(u)
	return prod