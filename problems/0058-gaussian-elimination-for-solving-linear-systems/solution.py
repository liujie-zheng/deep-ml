import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	b = b.reshape(-1, 1)
	aug = np.concatenate([A, b], axis=1)
	aug.astype(float)
	m, n = A.shape
	pivot_row, col = 0, 0
	while pivot_row < m and col < n:
		max_row = pivot_row + np.argmax(aug[pivot_row:, col])
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
	
	sol = np.zeros(m)
	for i in range(m - 1, -1, -1):
		last = aug[i, -1]
		for j in range(m - 1, i, -1):
			last -= aug[i, j] * sol[j]
		x_i = last / aug[i, i]
		sol[i] = x_i
	return sol