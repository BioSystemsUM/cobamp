from numpy import abs, where, compress, concatenate, ones, array, random, sign
from numpy.linalg import svd

EPSILON = 1e-10
PRECISION = 1e-10


def compute_nullspace(A, eps=1e-9, left=True):
	"""
	Computes the nullspace of the matrix A.

	Parameters

	----------

		A: A 2D-ndarray

		eps: Tolerance value for 0

		left: A boolean value indicating whether the result is the left nullspace (right if False)

	Returns the nullspace of A as a 2D ndarray
	-------

	"""
	u, s, v = svd(A)
	padding = max(0, A.shape[1] - s.shape[0])
	mask = concatenate(((s <= eps), ones((padding,), dtype=bool)), axis=0)
	return compress(mask, u.T, 0) if left else compress(mask, v.T, 1)


def nullspace_blocked_reactions(K, tolerance):
	"""

	Parameters

	----------

		K: A nullspace matrix as a 2D ndarray

		tolerance: Tolerance value for 0

	-------

	Returns indices of the rows of K where all values are 0


	"""
	return where(sum(abs(K.T) < tolerance) == K.shape[0])[0]


if __name__ == '__main__':
	import profile
	#
	# A = array([[2, 3, 5], [-4, 2, 3], [0, 0, 0]])
	# nullspace = compute_nullspace(A, left=False)

	Ar = random.rand(5000,10000) - 0.5
	Ar[(-0.4 < Ar) & (0.4 > Ar)] = 0
	A = sign(Ar)
	profile.run('nullspace = compute_nullspace(A, left=False)')