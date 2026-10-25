import numpy as np
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    # Your code here
    x = np.array(x)
    min_ = np.min(x)
    max_ = np.max(x)
    normed = (x - min_) / (max_ - min_)
    return normed.tolist()