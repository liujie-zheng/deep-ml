import torch

def empirical_pmf(samples: torch.Tensor) -> list:
    """
    Given a 1D tensor of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    samples = torch.tensor(samples)
    vals, counts = torch.unique(samples, return_counts=True)
    vals = vals.tolist()
    probs = (counts / len(samples)).tolist()
    return list(zip(vals, probs))