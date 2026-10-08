import torch

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient (float)
		- direction: Unit vector (torch.Tensor) in direction of steepest ascent
		- descent_direction: Unit vector (torch.Tensor) in direction of steepest descent
	"""
	# Your code here
	gradient = torch.tensor(gradient)
	mag = torch.linalg.vector_norm(gradient)
	if mag == 0:
		return {'magnitude': mag, 'direction': torch.zeros(gradient.shape[0]), 'descent_direction': torch.zeros(gradient.shape[0])}

	direc = gradient / mag
	desc = -direc

	return {'magnitude': mag, 'direction': direc, 'descent_direction': desc}

	