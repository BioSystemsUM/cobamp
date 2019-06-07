import abc
from cobamp.nullspace.subset_reduction import subset_reduction
from cobamp.utilities.property_management import PropertyDictionary
from cobamp.core.models import ConstraintBasedModel
from numpy import array, nonzero, ndarray
from itertools import chain, product


class ModelTransformer(object):
	__metaclass__ = abc.ABCMeta

	def transform(self, args, properties):

		# args must be:
		# - a dict with 'S', 'lb', 'ub' keys
		# - a ConstraintBasedModel
		if isinstance(args, dict):
			assert len(set(args.keys()) & {'S', 'lb', 'ub'}) == len(set(args.keys())), 'args must contain at least S' + \
																					   ', lb, and ub key-value pairs'

			S, lb, ub = [args[k] for k in ['S', 'lb', 'ub']]
			return self.transform_array(S, lb, ub, properties)

		elif isinstance(args, ConstraintBasedModel):
			S = args.get_stoichiometric_matrix()
			lb, ub = args.get_bounds_as_list()

			Sn, lbn, ubn, mapping, metabs = self.transform_array(S, lb, ub, properties)

			reaction_names_new = [properties['reaction_id_sep'].join([args.reaction_names[i] for i in mapping.from_new(i)]) for i in
								  range(len(lbn))]
			modeln = ConstraintBasedModel(
				S=Sn,
				thermodynamic_constraints=[list(k) for k in list(zip(lbn, ubn))],
				reaction_names=reaction_names_new,
				metabolite_names= [args.metabolite_names[k] for k in metabs]
			)

			return modeln, mapping, metabs

	@abc.abstractmethod
	def transform_array(self, S, lb, ub, properties):
		## TODO: implement

		# must return:
		# - new S matrix
		# - new lower/upper bounds
		# - mapping between rows/cols from both matrices
		# mapping = ReactionIndexMapping({}, {})
		# metabs = []
		# return S, lb, ub, mapping, metabs
		return


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
	def __init__(self, keep=None, block=None, absolute_bounds=False, reaction_id_sep='_+_'):

		def is_list(x):
			return isinstance(x, (tuple, list, ndarray))

		new_optional = {
			'keep': lambda x: is_list(x) or type(x) == None,
			'block': lambda x: is_list(x) or type(x) == None,
			'absolute_bounds': bool,
			'reaction_id_sep': str
		}

		super().__init__(optional_properties=new_optional)
		for name, value in zip(['keep', 'block', 'absolute_bounds', 'reaction_id_sep'],
							   [keep, block, absolute_bounds, reaction_id_sep]):
			self.add_if_not_none(name, value)


class SubsetReducer(ModelTransformer):
	TO_KEEP_SINGLE = 'SUBSET_REDUCER-TO_KEEP_SINGLE'
	TO_BLOCK = 'SUBSET_REDUCER-TO_BLOCK'
	ABSOLUTE_BOUNDS = 'SUBSET_REDUCER-ABSOLUTE_BOUNDS'

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

		return rd, nlb, nub, mapping, rdind

	def transform_array(self, S, lb, ub, properties):
		k, b, a = (properties[k] for k in ['keep', 'block', 'absolute_bounds'])

		Sn, lbn, ubn, mapping, metabs = self.reduce(S, lb, ub, k, b, a)

		return Sn, lbn, ubn, mapping, metabs

	def get_transform_maps(self, sub):
		new_to_orig = {i: list(nonzero(sub[i, :])[0]) for i in range(sub.shape[0])}
		orig_to_new = dict(chain(*[[(i, k) for i in v] for k, v in new_to_orig.items()]))

		return ReactionIndexMapping(orig_to_new, new_to_orig)
