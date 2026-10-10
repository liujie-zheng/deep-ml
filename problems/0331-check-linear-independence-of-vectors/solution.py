import torch

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Your code here
    vectors = torch.as_tensor(vectors, dtype=torch.float32)
    if len(vectors) == 0:
        return True
    m = vectors.shape[0]
    rank = torch.linalg.matrix_rank(vectors)
    return (rank == m).item()