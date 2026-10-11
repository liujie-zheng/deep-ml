import torch

def calculate_covariance_matrix(vectors) -> torch.Tensor:
    """
    Calculate the covariance matrix for given feature vectors using PyTorch.
    Input: 2D array-like of shape (n_features, n_observations).
    Returns a tensor of shape (n_features, n_features).
    """
    v_t = torch.as_tensor(vectors, dtype=torch.float)
    # Your implementation here
	means = torch.mean(v_t, axis=1, keepdims=True) # (2, 1)
	deviations = v_t - means # (2, 3)
	sums = deviations @ deviations.T
	cov = sums / (v_t.shape[1] - 1)
	return cov
