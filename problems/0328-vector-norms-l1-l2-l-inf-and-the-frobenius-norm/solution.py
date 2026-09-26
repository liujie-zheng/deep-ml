import torch

def compute_norm(arr: torch.Tensor, norm_type: str) -> float:
    """
    Compute the specified norm of the input tensor.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D tensor.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input tensor (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    if norm_type == "l1":
        return torch.sum(torch.abs(arr)).to(float).item()
    elif norm_type == "l2":
        square = torch.pow(arr, 2)
        return torch.sqrt(torch.sum(square)).to(float).item()
    elif norm_type == "linf":
        return torch.max(torch.abs(arr)).to(float).item()
    elif norm_type == "frobenius":
        if len(arr.shape) != 2:
            raise ValueError()
        else:
            square = torch.pow(arr, 2)
            return torch.sqrt(torch.sum(square)).to(float).item()
    else:
        raise ValueError()
