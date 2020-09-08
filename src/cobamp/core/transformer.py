import abc
from copy import deepcopy
from itertools import product

from cobamp.core.models import ConstraintBasedModel


class ModelTransformer(object):
	__metaclass__ = abc.ABCMeta

	@staticmethod
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
			new_properties = deepcopy(properties)

			for k in ['block', 'keep']:
				if new_properties[k] != None:
					new_properties[k] = [args.decode_index(r, 'reaction') for r in properties[k]]

			Sn, lbn, ubn, mapping, metabs = self.transform_array(S, lb, ub, new_properties)

			reaction_names_new = [
				new_properties['reaction_id_sep'].join([args.reaction_names[i] for i in mapping.from_new(i)]) for i in
				range(len(lbn))]
			modeln = ConstraintBasedModel(
				S=Sn,
				thermodynamic_constraints=[list(k) for k in list(zip(lbn, ubn))],
				reaction_names=reaction_names_new,
				metabolite_names=[args.metabolite_names[k] for k in metabs]
			)

			return modeln, mapping, metabs

	@staticmethod
	@abc.abstractmethod
	def transform_array(S, lb, ub, properties):
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
