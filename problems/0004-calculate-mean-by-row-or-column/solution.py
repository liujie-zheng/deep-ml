import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	matrix = np.array(matrix)
	if mode == 'row':
		return np.mean(matrix, axis=1)
	elif mode == 'column':
		return np.mean(matrix, axis=0)