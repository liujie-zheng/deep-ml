import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	# Your code here
	coff = 1 / (std_dev * math.sqrt(2 * math.pi))
	exp = math.exp(-(1/2) * ((x - mean) / std_dev) ** 2)
	val = coff * exp
	return round(val,5)