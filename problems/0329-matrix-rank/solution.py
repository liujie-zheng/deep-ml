import numpy as np

def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the rank of a matrix.
    
    Args:
        A: Input matrix of shape (m, n)
        tol: Tolerance for considering values as zero
    
    Returns:
        The rank of the matrix (integer)
    """
    # Your code here
    A = A.astype(float)
    m, n = A.shape
    pivot_row, col = 0, 0
    while pivot_row < m and col < n:
        max_row = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))
        if abs(A[max_row, col]) <= tol:
            col += 1
            continue
        
        A[pivot_row,:], A[max_row,:] = A[max_row,:].copy(), A[pivot_row,:].copy()

        for row in range(pivot_row + 1, m):
            factor = A[row, col] / A[pivot_row, col]
            to_substract = A[pivot_row, :] * factor
            A[row, :] -= to_substract
        pivot_row += 1
        col += 1
    
    return pivot_row