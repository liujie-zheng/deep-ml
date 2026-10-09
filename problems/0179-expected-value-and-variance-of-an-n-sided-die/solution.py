import torch

def dice_statistics(n: int) -> tuple[float, float]:
    """
    Compute the expected value and variance of a fair n-sided die roll using PyTorch.

    Args:
        n (int): Number of sides of the die

    Returns:
        tuple: (expected_value, variance)
    """
    # Your code here
    space = torch.arange(1, n + 1, dtype=torch.float32)
	mean = torch.mean(space)
	var = torch.var(space, unbiased=False)
	return (mean.item(), var.item())