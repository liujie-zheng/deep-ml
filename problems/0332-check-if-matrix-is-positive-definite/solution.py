import numpy as np

def check_positive_definite(matrix: list) -> dict:
    """
    Check if a matrix is positive definite and compute its eigenvalues.
    
    Args:
        matrix: A 2D list representing a square matrix
        
    Returns:
        dict with 'is_positive_definite' (bool) and 'eigenvalues' (list of floats sorted ascending)
    """
    # Your code here
    A = np.array(matrix)
    eigenvalues, _ = np.linalg.eig(A)
    is_pd = True
    for e in eigenvalues:
        if e <= 1e-10:
            is_pd = False
    return {'is_positive_definite': is_pd, 'eigenvalues': np.sort(np.round(eigenvalues, 4)).tolist()}