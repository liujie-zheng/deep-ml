import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    # Your implementation here
    if distribution == "uniform":
        samples = np.random.uniform(0, 1, size=(runs, n))
        means = np.mean(samples, axis=1)
        Z = (means - 0.5) / (np.sqrt(1/12) / np.sqrt(n))
    elif distribution == "exponential":
        samples = np.random.exponential(1.0, size=(runs, n))
        means = np.mean(samples, axis=1)
        Z = (means - 1) / (1 / np.sqrt(n))
    elif distribution == "bernoulli":
        samples = (np.random.rand(runs, n) < 0.3).astype(float)
        means = np.mean(samples, axis=1)
        Z = (means - 0.3) / (np.sqrt(0.3 * 0.7) / np.sqrt(n))

    Z_mean = np.mean(Z)
    Z_std = np.std(Z)
    return {'mean': Z_mean, 'std': Z_std}
