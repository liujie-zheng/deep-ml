import numpy as np
def dice_statistics(n: int) -> tuple[float, float]:
	"""
	Compute the expected value and variance of a fair n-sided die roll.

	Args:
		n (int): Number of sides of the die

	Returns:
		tuple: (expected_value, variance)
	"""
	# Your code here
	space = np.arange(1, n + 1)
	mean = np.mean(space)
	var = np.var(space)
	return (mean, var)