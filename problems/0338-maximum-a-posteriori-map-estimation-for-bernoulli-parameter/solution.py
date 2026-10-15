import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    observations = np.array(observations)
    k = np.sum(observations)
    n = len(observations)

    alpha_post = alpha + k
    beta_post = beta + n - k

    if alpha_post > 1 and beta_post > 1:
        theta_MAP = (alpha_post - 1) / (alpha_post + beta_post - 2)
    elif alpha_post == 1 and beta_post > 1:
        theta_MAP = 0.0
    elif alpha_post > 1 and beta_post == 1:
        theta_MAP = 1.0
    elif alpha_post == 1 and beta_post == 1:
        theta_MAP = 0.5
    return round(float(theta_MAP), 4)
    