import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    eigvals = np.linalg.eigvals(hessian)
    n = len(eigvals)
    pos, neg = 0, 0
    for val in eigvals:
        if abs(val) <= tol:
            return None
        if val > 0:
            pos += 1
        elif val < 0:
            neg += 1
    
    if pos == n:
        return -1
    if neg == n:
        return 1
    else:
        return 0