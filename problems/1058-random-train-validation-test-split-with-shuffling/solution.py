import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n = data.shape[0]
    shuffled = np.random.default_rng(seed).permutation(data)
    train_end = int(n * train_frac)
    validation_end = train_end + int(n * validation_frac)

    train_set = shuffled[:train_end, :]
    val_set = shuffled[train_end:validation_end, :]
    test_set = shuffled[validation_end:, :]

    return [train_set, val_set, test_set]