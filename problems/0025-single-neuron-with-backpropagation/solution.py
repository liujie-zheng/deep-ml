import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	bias = initial_bias
	weights = initial_weights.reshape(-1, 1)
	y = labels.reshape(-1, 1)
	mse_values = []

	for _ in range(epochs):
		# calculate y_pred
		z = features @ weights + bias # (3, 2) (2, 1) = (3, 1)
		y_pred = 1 / (1 + np.exp(-z))

		# calculate loss
		n = len(features)
		L_mse = np.mean((y_pred - y) ** 2)
		mse_values.append(round(float(L_mse), 4))

		dL_dz = (2 / n) * (y_pred - y) * y_pred * (1 - y_pred)

		Lw = features.T @ dL_dz # (2, 1)
		Lb = np.sum(dL_dz)

		weights = weights - learning_rate * Lw
		bias = bias - learning_rate * Lb
	
	return np.round(weights.flatten(), 4), round(float(bias), 4), mse_values