import numpy as np


def argmax_rand(arr):
	# np.argmax with random tie breaking
	return np.random.choice(np.flatnonzero(np.isclose(arr, np.max(arr), atol=1e-3)))