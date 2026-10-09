import torch

def descriptive_statistics(data) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset using PyTorch.
    
    Args:
        data: List, torch.Tensor, or array-like of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Your code here
    data = torch.tensor(data)
    data, _ = torch.sort(data)
    mean = torch.mean(data)
    median = torch.quantile(data, 0.5)
    vals, counts = torch.unique(data, return_counts=True)
    mode = vals[torch.argmax(counts)]
    var = torch.var(data, unbiased=False)
    std = torch.std(data, unbiased=False)
    p25 = torch.quantile(data, 0.25)
    p50 = torch.quantile(data, 0.5)
    p75 = torch.quantile(data, 0.75)
    inter_range = p75 - p25
    return {'mean': mean.item(), 'median': median.item(), 'mode': mode.item(), 'variance': var.item(), 'standard_deviation': std.item(), '25th_percentile': p25.item(), '50th_percentile': p50.item(), '75th_percentile': p75.item(), 'interquartile_range': inter_range.item()}