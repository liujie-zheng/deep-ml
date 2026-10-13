def conditional_probability(data, x, y):
    """
    Returns the probability P(Y=y|X=x) from list of (X, Y) pairs.
    Args:
      data: List of (X, Y) tuples
      x: value of X to condition on
      y: value of Y to check
    Returns:
      float: conditional probability, rounded to 4 decimal places
    """
    # Your code here
    x_count = 0
    y_count_given_x = 0
    for cx, cy in data:
      if cx == x:
        x_count += 1
        if cy == y:
          y_count_given_x += 1
    if x_count == 0:
      return 0
    return y_count_given_x / x_count