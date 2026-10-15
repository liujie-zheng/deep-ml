import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	# Your code here


	P = np.array(P)
	mask = P > 0
	logP = np.log(P[mask])
	HP = -np.sum(P * logP)

	Q = np.array(Q)
	logQ = np.log(Q[mask])
	HPQ = -np.sum(P * logQ)

	return (HP, HPQ)