import torch
import torch.nn.functional as F

def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate the cosine similarity of two vectors using PyTorch.
    Args:
        v1 (torch.Tensor): 1D tensor representing the first vector.
        v2 (torch.Tensor): 1D tensor representing the second vector.
    Returns:
        float: The cosine similarity of the two vectors.
    """
    # Implement your code here
    sim = (v1 @ v2) / (torch.linalg.norm(v1.to(float)) * torch.linalg.norm(v2.to(float)))
    return sim.item()