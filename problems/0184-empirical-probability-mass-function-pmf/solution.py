import numpy as np
def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    # TODO: Implement the function
    samples = np.array(samples)
    vals, counts = np.unique(samples, return_counts=True)
    return list(zip(vals, counts / len(samples)))