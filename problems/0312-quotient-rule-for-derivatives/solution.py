import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    # Your code here
    ng = len(g_coeffs)
    nh = len(h_coeffs)

    gx = 0
    for i in range(ng):
        gx += g_coeffs[i] * (x ** (ng - 1 - i))
    
    hx = 0
    for i in range(nh):
        hx += h_coeffs[i] * (x ** (nh -1 - i))

    dg = []
    for i in range(ng):
        dg.append(g_coeffs[i] * (ng - 1 - i))
    
    dh = []
    for i in range(nh):
        dh.append(h_coeffs[i] * (nh - 1 - i))

    dgx = 0
    for i in range(ng - 1):
        dgx += dg[i] * (x ** (ng - 2 - i))
    
    dhx = 0
    for i in range(nh - 1):
        dhx += dh[i] * (x ** (nh - 2 - i))
    
    return (hx * dgx - gx * dhx) / (hx ** 2)