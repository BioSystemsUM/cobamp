import abc
from cobamp.nullspace.subset_reduction import subset_reduction
from cobamp.utilities.property_management import PropertyDictionary
from numpy import array, nonzero, ndarray
from itertools import chain, product


class ModelTransformer(abc.ABCMeta):
	@abc.abstractmethod
	def transform(self, S, lb, ub, properties):
		## TODO: implement
		# must return:
		# - new S matrix
		# - new lower/upper bounds
		# - mapping between rows/cols from both matrices
		pass


class ReactionIndexMapping(object):
	def __init__(self, otn, nto):
		self.otn = otn
		self.nto = nto

	def from_original(self, idx):
		return self.otn[idx]

	def from_new(self, idx):
		return self.nto[idx]

	def multiply(self, new_ids):
		return list(product(*[self.from_new(k) for k in set(new_ids)]))


class SubsetReducerProperties(PropertyDictionary):
	def __init__(self, keep=None, block=None, absolute_bounds=False):
		def is_list(x):
			return isinstance(x, (tuple, list, ndarray))

		new_optional = {
			'keep': lambda x: is_list(x) or type(x) == None,
			'block': lambda x: is_list(x) or type(x) == None,
			'absolute_bounds': bool
		}

		super().__init__(optional_properties=new_optional)
		for name, value in zip(['keep', 'block', 'absolute_bounds'], [keep, block, absolute_bounds]):
			self.add_if_not_none(name, value)


class SubsetReducer(object):
	TO_KEEP_SINGLE = 'SUBSET_REDUCER-TO_KEEP_SINGLE'
	TO_BLOCK = 'SUBSET_REDUCER-TO_BLOCK'
	ABSOLUTE_BOUNDS = 'SUBSET_REDUCER-ABSOLUTE_BOUNDS'

	def __init__(self):
		pass

	def reduce(self, S, lb, ub, keep=(), block=(), absolute_bounds=False):
		lb, ub = list(map(array, [lb, ub]))
		to_keep, to_block = [], []
		irrev = (lb >= 0) | (ub <= 0) & (lb <= 0)

		if block:
			to_block = array(block)

		if keep:
			to_keep = array(keep)

		rd, sub, irrev_reduced, rdind, irrv_subsets, kept_reactions, K, _ = subset_reduction(
			S, irrev, to_keep_single=to_keep, to_remove=to_block)

		mapping = self.get_transform_maps(sub)

		nlb = [0 if irrev_reduced[k] else None for k in range(rd.shape[1])]
		nub = [None] * rd.shape[1]

		if absolute_bounds:
			nlb = [0 if irrev_reduced[k] else -float('inf') for k in range(rd.shape[1])]
			nub = [float('inf')] * rd.shape[1]
			alb, aub = list(zip(*[[fx([x[k] for k in mapping.from_new(i)]) for x, fx in zip([lb, ub], [max, min])]
								  for i in range(rd.shape[1])]))

			for func, pair in zip([max, min], [[nlb, alb], [nub, aub]]):
				new, absolute = pair
				for i, v in enumerate(absolute):
					new[i] = func(new[i], absolute[i])

		return rd, nlb, nub, mapping

	def transform(self, S, lb, ub, properties):
		k, b, a = (properties[k] for k in ['keep', 'block', 'absolute_bounds'])

		return self.reduce(S, lb, ub, k, b, a)

	def get_transform_maps(self, sub):
		new_to_orig = {i: list(nonzero(sub[i, :])[0]) for i in range(sub.shape[0])}
		orig_to_new = dict(chain(*[[(i, k) for i in v] for k, v in new_to_orig.items()]))

		return ReactionIndexMapping(orig_to_new, new_to_orig)
