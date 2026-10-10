import torch

def gaussian_elimination(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.

    :param A: Coefficient matrix (torch.Tensor)
    :param b: Right-hand side vector (torch.Tensor)
    :return: Solution vector x (torch.Tensor)
    """
    b = b.reshape(-1, 1)
	aug = torch.concatenate([A, b], axis=1)
	aug.to(torch.float32)
	m, n = A.shape
	pivot_row, col = 0, 0
	while pivot_row < m and col < n:
		max_row = pivot_row + torch.argmax(aug[pivot_row:, col]).item()
		if aug[max_row, col] == 0:
			col += 1
			continue
		
		if max_row != pivot_row:
			aug[[pivot_row, max_row]] = aug[[max_row, pivot_row]]
		
		for row in range(pivot_row + 1, m):
			factor = aug[row, col] / aug[pivot_row, col]
			to_sub = factor * aug[pivot_row, :]
			aug[row, :] -= to_sub
		
		pivot_row += 1
		col += 1
	
	sol = torch.zeros(m)
	for i in range(m - 1, -1, -1):
		last = aug[i, -1]
		for j in range(m - 1, i, -1):
			last -= aug[i, j] * sol[j]
		x_i = last / aug[i, i]
		sol[i] = x_i
	return sol