import torch

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> torch.Tensor:
    """
    Compute the Jacobian matrix using numerical differentiation.
    
    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences
    
    Returns:
        Jacobian matrix as torch.Tensor
    """
    # Your code here
    f_x = torch.tensor(f(x), dtype=torch.float64) # (m, )
	m = len(f_x)
	n = len(x)

	matrix = torch.zeros((m, n))
	
	for j in range(n):
		xp = x.copy()
		xp[j] += h # (n, )
		f_xp = torch.tensor(f(xp), dtype=torch.float64)
		diff = (f_xp - f_x) / h # (m, )
		matrix[:, j] = diff # (m, 1)

	return matrix