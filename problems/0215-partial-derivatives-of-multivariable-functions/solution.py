import numpy as np

def compute_partial_derivatives(func_name: str, point: tuple[float, ...]) -> tuple[float, ...]:
	"""
	Compute partial derivatives of multivariable functions.
	
	Args:
		func_name: Function identifier
			'poly2d': f(x,y) = x²y + xy²
			'exp_sum': f(x,y) = e^(x+y)
			'product_sin': f(x,y) = x·sin(y)
			'poly3d': f(x,y,z) = x²y + yz²
			'squared_error': f(x,y) = (x-y)²
		point: Point (x, y) or (x, y, z) at which to evaluate
	
	Returns:
		Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
	"""
	# Your code here
	if func_name == 'poly2d':
		# df/dx = 2xy + y^2
		# df/dy = x^2 + 2xy
		x = point[0]
		y = point[1]
		df_dx = 2 * x * y + y ** 2
		df_dy = x ** 2 + 2 * x * y
		return (df_dx, df_dy)
	elif func_name == 'exp_sum':
		# df/dx = e^(x+y)
		# df/dy = e^(x+y)
		x = point[0]
		y = point[1]
		df_dx = np.exp(x + y)
		df_dy = np.exp(x + y)
		return (df_dx, df_dy)
	elif func_name == 'product_sin':
		# df/dx = sin(y)
		# df/dy = x cos(y)
		x = point[0]
		y = point[1]
		df_dx = np.sin(y)
		df_dy = np.cos(y) * x
		return (df_dx, df_dy)
	elif func_name == 'poly3d':
		# df/dx = 2xy
		# df/dy = x^2 + z^2
		# df/dz = 2zy
		x = point[0]
		y = point[1]
		z = point[2]
		df_dx = 2 * x * y
		df_dy = x ** 2 + z ** 2
		df_dz = 2 * z * y
		return (df_dx, df_dy, df_dz)
	elif func_name == 'squared_error':
		# df/dx = 2(x - y)
		# df/dy = -2(x - y)
		x = point[0]
		y = point[1]
		df_dx = 2 * (x - y)
		df_dy = -2 * (x - y)
		return (df_dx, df_dy)