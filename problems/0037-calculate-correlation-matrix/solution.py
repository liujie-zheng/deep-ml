import torch
from typing import Optional, Union

def calculate_correlation_matrix(
    X: Union[torch.Tensor, list, "np.ndarray"],
    Y: Optional[Union[torch.Tensor, list, "np.ndarray"]] = None
) -> torch.Tensor:
    """
    Compute the correlation matrix of X (and optionally Y) using PyTorch.
    If Y is None, returns the correlation matrix of X with itself.
    """
    # Your implementation here
    X = torch.as_tensor(X, dtype=torch.float32)
    if Y is None:
		Y = X
    else:
        Y = torch.as_tensor(Y, dtype=torch.float32)
    
	
	dev_x = X - torch.mean(X, axis=0) # (3, 2)
	dev_y = Y - torch.mean(Y, axis=0)

	cov = (1 / X.shape[0]) * (dev_x.T @ dev_y) #(2, 2)

	std_x = torch.std(X, axis=0, keepdims=True, correction=0) # (1, 2)
	std_y = torch.std(Y, axis=0, keepdims=True, correction=0)

	return cov / (std_x.T @ std_y)
