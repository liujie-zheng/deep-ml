import torch

def matrix_determinant_and_trace(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the determinant and trace of a square matrix.
    
    Args:
        matrix: A square matrix (n x n) as a torch.Tensor
    
    Returns:
        Tuple of (determinant, trace) as torch.Tensors
    """
    # Your code here
    matrix = torch.tensor(matrix)
	det = torch.linalg.det(matrix)
	n = matrix.shape[0]
	trace = torch.sum(torch.eye(n) * matrix)
	return (det, trace)