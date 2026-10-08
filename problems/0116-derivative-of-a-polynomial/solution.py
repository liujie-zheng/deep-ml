def poly_term_derivative(c: float, x: float, n: float) -> float:
    # Your code here
    coefficient = c * n
    power = n - 1
    return coefficient * (x ** power)