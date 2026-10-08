import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	# Your code here
	gradient = np.array(gradient)
	mag = np.linalg.norm(gradient)
	if mag == 0:
		return {'magnitude': mag, 'direction': [0] * gradient.shape[0], 'descent_direction': [0] * gradient.shape[0]}

	direc = gradient / mag
	desc = -direc

	return {'magnitude': mag, 'direction': direc.tolist(), 'descent_direction': desc.tolist()}