import numpy as np

def to_categorical(x, n_col=None):
	# Your code here
	x = np.array(x)
	if n_col is None:
		n_col = np.max(x) + 1
	n_row = len(x)
	one_hot = np.zeros((n_row, n_col))

	for i in range(len(x)):
		one_hot[i, x[i]] = 1
	
	return one_hot