import torch

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> torch.Tensor:
    """
    Compute partial derivatives of multivariable functions using PyTorch autograd.
    
    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x^2*y + x*y^2
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x*sin(y)
            'poly3d': f(x,y,z) = x^2*y + y*z^2
            'squared_error': f(x,y) = (x-y)^2
        point: Point (x, y) or (x, y, z) at which to evaluate
    
    Returns:
        Tensor of partial derivatives [df/dx, df/dy, ...] at point
    """
    # Your code here
    if func_name == 'poly2d':
		# df/dx = 2xy + y^2
		# df/dy = x^2 + 2xy
		x = torch.tensor(point[0])
		y = torch.tensor(point[1])
		df_dx = 2 * x * y + y ** 2
		df_dy = x ** 2 + 2 * x * y
		return (df_dx, df_dy)
	elif func_name == 'exp_sum':
		# df/dx = e^(x+y)
		# df/dy = e^(x+y)
		x = torch.tensor(point[0])
		y = torch.tensor(point[1])
		df_dx = torch.exp(x + y)
		df_dy = torch.exp(x + y)
		return (df_dx, df_dy)
	elif func_name == 'product_sin':
		# df/dx = sin(y)
		# df/dy = x cos(y)
		x = torch.tensor(point[0])
		y = torch.tensor(point[1])
		df_dx = torch.sin(y)
		df_dy = torch.cos(y) * x
		return (df_dx, df_dy)
	elif func_name == 'poly3d':
		# df/dx = 2xy
		# df/dy = x^2 + z^2
		# df/dz = 2zy
		x = torch.tensor(point[0])
		y = torch.tensor(point[1])
		z = torch.tensor(point[2])
		df_dx = 2 * x * y
		df_dy = x ** 2 + z ** 2
		df_dz = 2 * z * y
		return (df_dx, df_dy, df_dz)
	elif func_name == 'squared_error':
		# df/dx = 2(x - y)
		# df/dy = -2(x - y)
		x = torch.tensor(point[0])
		y = torch.tensor(point[1])
		df_dx = 2 * (x - y)
		df_dy = -2 * (x - y)
		return (df_dx, df_dy)