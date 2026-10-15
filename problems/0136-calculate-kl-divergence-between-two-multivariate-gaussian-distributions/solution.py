import numpy as np

def multivariate_kl_divergence(mu_p: np.ndarray, Cov_p: np.ndarray, mu_q: np.ndarray, Cov_q: np.ndarray) -> float:
    """
    Computes the KL divergence between two multivariate Gaussian distributions.
    
    Parameters:
    mu_p: mean vector of the first distribution
    Cov_p: covariance matrix of the first distribution
    mu_q: mean vector of the second distribution
    Cov_q: covariance matrix of the second distribution

    Returns:
    KL divergence as a float
    """
    # Your code here
    k = len(mu_p)

    Cov_q_inv = np.linalg.inv(Cov_q)

    Cov_q_inv_Cov_p = Cov_q_inv @ Cov_p
    trace = np.trace(Cov_q_inv_Cov_p)

    delta_mu = mu_q - mu_p
    mah_dist = delta_mu.T @ Cov_q_inv @ delta_mu

    log_det = np.log(np.linalg.det(Cov_q)) - np.log(np.linalg.det(Cov_p))

    kl = (1/2) * (trace + mah_dist - k + log_det)
    return float(kl)