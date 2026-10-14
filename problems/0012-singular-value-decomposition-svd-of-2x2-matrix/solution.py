import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    # Your code here
    S = A.T @ A
    theta = (1/2) * np.arctan2(2*S[0,1], S[0,0] - S[1,1])
    c = np.cos(theta)
    s = np.sin(theta)
    V = np.array([[c, -s], [s, c]])
    B = A @ V
    sigma_1 = np.linalg.norm(B[:,0])
    sigma_2 = np.linalg.norm(B[:,1])
    if sigma_1 < sigma_2:
        sigma_1, sigma_2 = sigma_2, sigma_1
        V = V[:, [1,0]]
    u_1 = B[:,0] / sigma_1
    u_2 = B[:,1] / sigma_2
    U = np.stack((u_1, u_2), axis=1)
    return (U, np.array([sigma_1, sigma_2]), V.T)