import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    x = np.array(x)
    n = len(x)
    numerical_grad = np.zeros_like(x)
    for i in range(n):
        xp1, xp2 = x.copy(), x.copy()
        xp1[i] += epsilon
        xp2[i] -= epsilon

        g = (f(xp1) - f(xp2)) / (2 * epsilon)
        numerical_grad[i] = g

    relative_error = np.linalg.norm(numerical_grad - analytical_grad) / (np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad))
    return (numerical_grad, relative_error)