from cobamp.algorithms.kshortest import InterventionProblem

import abc
import numpy as np

MAX_PRECISION = 1e-10


class AbstractObjectReader(object):
	"""
	An abstract class for reading metabolic model objects from external frameworks, and extracting the data needed for
	pathway analysis methods. Also deals with name conversions.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, model):
		"""
		Parameters

		----------

			model: A Model instance from the external framework to use. Must be registered in the dict stored as
			external_wrappers.model_readers along with its reader.

		"""
		self.model = model
		self.r_ids, self.m_ids = self.get_metabolite_and_reactions_ids()
		self.rx_instances = self.get_rx_instances()
		self.S = self.get_stoichiometric_matrix()
		self.irrev_bool = self.get_irreversibilities(False)
		self.irrev_index = self.get_irreversibilities(True)
		self.lb, self.ub = tuple(zip(*self.get_model_bounds(False)))
		self.bounds_dict = self.get_model_bounds(True)

	@abc.abstractmethod
	def get_stoichiometric_matrix(self):
		"""
		Returns a 2D numpy array with the stoichiometric matrix whose metabolite and reaction indexes match the names
		defined in the class variables r_ids and m_ids
		"""
		return

	@abc.abstractmethod
	def get_model_bounds(self, as_dict, separate_list):
		"""
		Returns the lower and upper bounds for all fluxes in the model. This either comes in the form of an ordered list
		with tuples of size 2 (lb,ub) or a dictionary with the same tuples mapped by strings with reaction identifiers.

		Parameters

		----------

			as_dict: A boolean value that controls whether the result is a dictionary mapping str to tuple of size 2
			separate: A boolean value that controls whether the result is two numpy.array(), one for lb and the other\n
			to ub
		"""
		return

	@abc.abstractmethod
	def get_irreversibilities(self, as_index):
		"""
		Returns a vector representing irreversible reactions, either as a vector of booleans (each value is a flux,
		ordered in the same way as reaction identifiers) or as a vector of reaction indexes.

		Parameters

		----------

			as_dict: A boolean value that controls whether the result is a vector of indexes

		"""
		return

	@abc.abstractmethod
	def get_rx_instances(self):
		"""
		Returns the reaction instances contained in the model. Varies depending on the framework.
		"""
		return

	@abc.abstractmethod
	def get_metabolite_and_reactions_ids(self):
		"""
		Returns two ordered iterables containing the metabolite and reaction ids respectively.
		"""
		return

	def reaction_id_to_index(self, id):
		"""
		Returns the numerical index of a reaction when given a string representing its identifier.

		Parameters

		----------

			id: A reaction identifier as a string

		"""
		return self.r_ids.index(id)

	def metabolite_id_to_index(self, id):
		"""
		Returns the numerical index of a metabolite when given a string representing its identifier.

		Parameters

		----------

			id: A metabolite identifier as a string

		"""
		return self.m_ids.index(id)

	def convert_constraint_ids(self, tup, yield_constraint):
		if yield_constraint:
			constraint = tuple(list(map(self.reaction_id_to_index, tup[:2])) + list(tup[2:]))
		else:
			constraint = tuple([self.reaction_id_to_index(tup[0])] + list(tup[1:]))
		return constraint

	def decode_k_shortest_solution(self, solarg):
		if isinstance(solarg, list):
			return [self.__decode_k_shortest_solution(sol) for sol in solarg]
		else:
			return self.__decode_k_shortest_solution(solarg)

	def __decode_k_shortest_solution(self, sol):
		## TODO: Make MAX_PRECISION a parameter for linear systems or the KShortestAlgorithm
		return {self.r_ids[k]: v for k, v in sol.attribute_value(sol.SIGNED_VALUE_MAP).items() if
				abs(v) > MAX_PRECISION}


class COBRAModelObjectReader(AbstractObjectReader):

	def get_stoichiometric_matrix(self):
		S = np.zeros((len(self.m_ids), len(self.r_ids)))
		for i, r_id in enumerate(self.r_ids):
			for metab, coef in self.model.reactions.get_by_id(r_id).metabolites.items():
				S[self.m_ids.index(metab.id), i] = coef

		return S

	def get_model_bounds(self, as_dict = False, separate_list=False):
		bounds = [r.bounds for r in self.rx_instances]
		if as_dict:
			return dict(zip(self.r_ids, bounds))
		else:
			if separate_list:
				return [bounds for bounds in list(zip(*tuple(bounds)))]
			else:
				return tuple(bounds)

	def get_irreversibilities(self, as_index):
		irrev = [not r.reversibility for r in self.rx_instances]
		if as_index:
			irrev = list(np.where(irrev)[0])
		return irrev

	def get_rx_instances(self):
		return [self.model.reactions.get_by_id(rx) for rx in self.r_ids]

	def get_metabolite_and_reactions_ids(self):
		return tuple([[x.id for x in lst] for lst in (self.model.reactions, self.model.metabolites)])

class FramedModelObjectReader(AbstractObjectReader):

	def get_stoichiometric_matrix(self):
		return np.array(self.model.stoichiometric_matrix())

	def get_model_bounds(self, as_dict=False, separate_list=False):
		bounds = [(r.lb, r.ub) for r in self.rx_instances]
		if as_dict:
			return dict(zip(self.r_ids, bounds))
		else:
			if separate_list:
				return [bounds for bounds in list(zip(*tuple(bounds)))]
			else:
				return tuple(bounds)

	def get_irreversibilities(self, as_index):
		irrev = [not r.reversible for r in self.rx_instances]
		if as_index:
			irrev = list(np.where(irrev)[0])
		return irrev

	def get_metabolite_and_reactions_ids(self):
		return tuple(self.model.reactions.keys()), tuple(self.model.metabolites.keys())

	def get_rx_instances(self):
		return [self.model.reactions[rx] for rx in self.r_ids]


class CobampModelObjectReader(AbstractObjectReader):

	def get_stoichiometric_matrix(self):
		return self.model.get_stoichiometric_matrix()

	def get_model_bounds(self, as_dict, separate_list = False):
		if as_dict:
			return dict(zip(self.r_ids, self.model.bounds))
		else:
			if separate_list:
				return [bounds for bounds in list(zip(*tuple(self.model.bounds)))]
			else:
				return tuple(self.model.bounds)

	def get_irreversibilities(self, as_index):
		irrev = [not self.model.is_reversible_reaction(r) for r in self.r_ids]
		if as_index:
			irrev = list(np.where(irrev)[0])
		return irrev

	def get_metabolite_and_reactions_ids(self):
		return tuple([[x.id for x in lst] for lst in (self.model.reactions, self.model.metabolites)])

	def get_rx_instances(self):
		return None


# This dict contains the mapping between model instance namespaces (without the class name itself) and the appropriate
# model reader object. Modify this if a new reader is implemented.

model_readers = {
	'cobra.core.model': COBRAModelObjectReader,
	'framed.model.cbmodel': FramedModelObjectReader,
	'cobamp.core.models': CobampModelObjectReader
}
