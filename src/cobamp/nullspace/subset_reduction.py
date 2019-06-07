"""
Inspired by Metatool's code
"""

from numpy import sqrt, triu, logical_not, nonzero, mean, zeros, argmin, isin, sign, append, delete, unique, where, \
	array, dot, eye
from numpy.linalg import norm

from ..nullspace.nullspace import compute_nullspace, nullspace_blocked_reactions

EPSILON = 2 ** -52
PRECISION = 1e-10


def subset_reduction(S, irrev, to_remove=[], to_keep_single=[]):
	"""
	Reduces a stoichiometric matrix using nullspace analysis by identifying linearly dependent (enzyme) subsets.
	These reactions are then compressed.

	Parameters
	----------
		S: Stoichiometric matrix as an ndarray.

		irrev: A boolean array with size equal to the number of columns in the S matrix.

		to_remove: A list of indices specifying columns of the S matrix to remove before the compression (usually blocked
		reactions)

		to_keep_single: A list of indices specifying columns of the S matrix not to compress.

	Returns	rd, sub, irrev_reduced, rdind, irrv_subsets, kept_reactions, kernel, correlation_matrix

		rd : compressed stoichiometric matrix -> numpy.array

		sub : subset matrix, n-subsets by n-reactions -> numpy.array

		irrev_reduced : subset reversibilities -> numpy.array of type bool

		rdind : metabolite indices -> numpy.array of type int

		irrv_subsets : same as sub, but list if empty

		kept_reactions : indexes for reactions used in the network compression

		kernel : numpy.array with the right nullspace of S

		correlation_matrix : numpy.array with reaction correlation matrix
	-------

	"""
	m, n = S.shape

	keep_single = array([False] * n)
	keep_single[to_keep_single] = True

	kept_reactions = array([True] * n)
	kept_reactions[to_remove] = False
	kept_reactions = where(kept_reactions)[0]

	ktol = EPSILON * sum(kept_reactions)

	kernel = compute_nullspace(S[:, kept_reactions], ktol, False)
	kernel_blocked = nullspace_blocked_reactions(kernel, ktol)

	if kernel_blocked.shape[0] > 0:
		kept_reactions = kept_reactions[kernel_blocked]
		kernel = compute_nullspace(S[:, kept_reactions], ktol, False)

	correlation_matrix = subset_candidates(kernel)
	S_scm = S[:, kept_reactions]
	irrev_scm = irrev[kept_reactions]
	scm_kp_ids = where([keep_single[kept_reactions]])[1]

	sub, irrev_reduced, irrv_subsets = subset_correlation_matrix(S_scm, kernel, irrev_scm, correlation_matrix,
																 scm_kp_ids)
	if len(kept_reactions) < n:
		temp = zeros([sub.shape[0], n])
		temp[:, kept_reactions] = sub
		sub = temp
		if len(irrv_subsets) > 0:
			temp = zeros([len(irrv_subsets), n])
			temp[:, kept_reactions] = irrv_subsets
			irrv_subsets = temp
	rd, rdind, dummy = reduce(S, sub, irrev_reduced)

	return rd, sub, irrev_reduced, rdind, irrv_subsets, kept_reactions, kernel, correlation_matrix


def subset_candidates(kernel, tol=None):
	"""
	Computes a matrix of subset candidates from the nullspace of the S matrix

	Parameters

	----------

		kernel: Nullspace of the S matrix

		tol: Tolerance to 0.

	Returns a 2D triangular ndarray
	-------

	"""
	tol = kernel.shape[0] * EPSILON if tol is None else tol
	cr = dot(kernel, kernel.T)
	for i in range(kernel.shape[0]):
		for j in range(i + 1, kernel.shape[0]):
			
			cr[i, j] = cr[i, j] / sqrt(cr[i, i] * cr[j, j])
		cr[i, i] = 1
	cr = triu(cr)
	cr[abs(abs(cr) - 1) >= tol] = 0
	return sign(cr)


def subset_correlation_matrix(S, kernel, irrev, cr, keepSingle=None):
	"""

	Parameters
	----------
		S: Stoichiometric matrix as ndarray

		kernel: The nullspace of S

		irrev: List of booleans representing irreversible reactions (when True)

		cr: The subset candidate matrix, computed using <subset_candidates>

		keepSingle: List of reaction indices that will not be compressed.

	Returns sub, irrev_sub, irrev_violating_subsets

		sub : subset matrix, n-subsets by n-reactions -> numpy.array

		irrev_sub : subset reversibilities -> numpy.array of type bool

		irrev_violating_subsets : same as sub, but list if empty. Contains subsets discarded due to irreversibility faults
	-------

	"""
	m, n = S.shape
	in_subset = array([False] * n)
	irrev_sub = array([False] * cr.shape[0])
	sub = zeros([cr.shape[0], n])
	irrev_violating_subsets = []
	sub_count = 0
	if (keepSingle is not None) and (len(keepSingle) > 0):
		# keepSingle = array([])
		irrev_violating_subsets = []
		sub[:len(keepSingle),keepSingle] = eye(len(keepSingle))
		irrev_sub[:len(keepSingle)] = irrev[keepSingle]
		in_subset[keepSingle] = True
		sub_count = len(keepSingle)
	for i in range(cr.shape[0] - 1, -1, -1):
		reactions = where(cr[:, i] != 0)[0]
		in_subset_indexes = where(in_subset)[0]
		in_subset_reactions = isin(reactions, in_subset_indexes)
		reactions = reactions[logical_not(in_subset_reactions)]
		if len(reactions) > 0:
			in_subset[reactions] = True
			irrev_sub[sub_count] = (irrev[reactions]).any()

			if len(reactions) == 1:
				sub[sub_count, reactions] = 1
			else:
				lengths = norm(kernel[reactions, :], axis=1)
				min_ind = argmin(abs(lengths - mean(lengths)))
				lengths /= lengths[min_ind]
				sub[sub_count, reactions] = lengths * cr[reactions, i]
			sub_count += 1

	sub = sub[:sub_count,:]
	irrev_sub = irrev_sub[:sub_count]

	ind = where(sub[:, irrev] < 0)[0]
	if len(ind) > 0:
		sub[ind, :] = -sub[ind, :]
		ind = where(sub[:, irrev] < 0)[0]
		if len(ind) > 0:
			irrev_violating_subsets = sub[ind, :]
			sub = delete(sub, ind, 0)
			irrv_to_keep = delete(array(range(len(irrev_sub))), ind, 0)
			irrev_sub = irrev_sub[irrv_to_keep]

	return sub, irrev_sub, irrev_violating_subsets


def reduce(S, sub, irrev_reduced=None):
	"""
	Reduces a stoichiometric matrix according to the subset information present in the sub matrix and irrev_reduced.

	Parameters

	----------

		S: Stoichiometric matrix

		sub: Subset matrix as computed by <subset_correlation_matrix>

		irrev_reduced: Irreversibility vector regarding the subsets.

	Returns reduced, reduced_indexes, irrev_reduced

	-------

	"""

	reduced = dot(S, sub.T)
	reduced[abs(reduced) < PRECISION] = 0
	reduced_indexes = unique(nonzero(reduced)[0])
	reduced = reduced[reduced_indexes, :]

	rdm, rdn = reduced.shape
	if rdn == 0 or rdm == 0:
		reduced = zeros(1, rdn)

	if irrev_reduced is not None:
		ind = unique(nonzero(reduced)[1])
		reduced = reduced[:, ind]
		irrev_reduced = irrev_reduced[ind]
	else:
		irrev_reduced = []

	return reduced, reduced_indexes, irrev_reduced
