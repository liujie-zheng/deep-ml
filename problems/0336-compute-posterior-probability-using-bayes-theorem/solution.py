import numpy as np
def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
	"""
	Calculate posterior probabilities using Bayes' Theorem.
	
	Args:
		priors: Prior probabilities P(H_i) for each hypothesis
		likelihoods: Likelihoods P(E|H_i) for each hypothesis
		
	Returns:
		Posterior probabilities P(H_i|E) for each hypothesis
	"""
	# P(H_i|E) = P(E|H_i)P(H_i) / P(E)
	P_Hi = np.array(priors)
	P_E_Hi = np.array(likelihoods)
	joint = P_E_Hi * P_Hi
	P_E = np.sum(joint)

	return joint / P_E