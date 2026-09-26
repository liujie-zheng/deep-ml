import torch

def vector_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor | int:
    # Return the element-wise sum of vectors 'a' and 'b'.
    # If vectors have different lengths, return -1.
    if a.shape[0] != b.shape[0]:
        return -1

    return a + b