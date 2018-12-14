from ..efm_enumeration.kshortest_efms import KShortestEFMAlgorithm
from ..linear_systems.linear_systems import IrreversibleLinearSystem, DualLinearSystem, IrreversibleLinearPatternSystem
from ..mcs_enumeration.intervention_problem import *

from ..efm_enumeration import kshortest_efm_properties as kp

import abc
import numpy as np


class KShortestEnumeratorWrapper(object):
	__metaclass__ = abc.ABCMeta
	"""
	An abstract class for methods involving the K-shortest EFM enumeration algorithm
	"""
	ALGORITHM_TYPE_ITERATIVE = 'kse_iterative'
	ALGORITHM_TYPE_POPULATE = 'kse_populate'

	__alg_to_prop_name = {
		ALGORITHM_TYPE_ITERATIVE: kp.K_SHORTEST_OPROPERTY_MAXSOLUTIONS,
		ALGORITHM_TYPE_POPULATE: kp.K_SHORTEST_OPROPERTY_MAXSIZE
	}

	__alg_to_alg_name = {
		ALGORITHM_TYPE_ITERATIVE: kp.K_SHORTEST_METHOD_ITERATE,
		ALGORITHM_TYPE_POPULATE: kp.K_SHORTEST_METHOD_POPULATE
	}

	def __init__(self, model, algorithm_type=ALGORITHM_TYPE_POPULATE, stop_criteria=1, forced_solutions=None,
				 excluded_solutions=None):
		"""

		Parameters

		----------

			model: A Model instance from the external framework to use. Must be registered in the dict stored as
			external_wrappers.model_readers along with its reader.

			algorithm_type: ALGORITHM_TYPE_ITERATIVE or ALGORITHM_TYPE_POPULATE constants stored as class attributes.
				ALGORITHM_TYPE_ITERATIVE is a slower method (regarding EFMs per unit of time) that enumerates EFMs one
				at a time.
				ALGORITHM_TYPE_POPULATE enumerates EFMs one size at a time. This is the preferred method as it's
				generally faster.

			stop_criteria: An integer that defines the stopping point for EFM enumeration. Either refers to the maximum
			number of EFMs or the maximum size they can reach before the enumeration stops.

			forced_solutions: A list of KShortestSolution or lists of reaction indexes that must show up in the
			enumeration process. (experimental feature)

			excluded_solutions: A list of KShortestSolution or lists of reaction indexes that cannot show up in the
			enumeration process. (experimental feature)
		"""

		self.__model = model
		if model.__module__ in model_readers.keys():
			self.model_reader = model_readers[model.__module__](model)
		else:
			raise TypeError(
				"The `model` instance is not currently supported by cobamp. Currently available readers are: " + str(
					list(model_readers.keys())))

		self.__algo_properties = kp.KShortestProperties()
		self.__algo_properties[kp.K_SHORTEST_MPROPERTY_METHOD] = self.__alg_to_alg_name[algorithm_type]
		self.__algo_properties[self.__alg_to_prop_name[algorithm_type]] = stop_criteria
		self.__forced_solutions = forced_solutions
		self.__excluded_solutions = excluded_solutions
		self.__setup_algorithm()
		self.enumerated_sols = []

	def __setup_algorithm(self):
		"""
		Creates the algorithm instance

		Returns:

		"""
		self.__algo = KShortestEFMAlgorithm(self.__algo_properties, False)

	def __get_forced_solutions(self):
		"""
		Returns: A list of KShortestSolution or lists of reaction indexes

		"""
		return self.__forced_solutions

	def __get_excluded_solutions(self):
		"""
		Returns: A list of KShortestSolution or lists of reaction indexes

		"""
		return self.__excluded_solutions

	@abc.abstractmethod
	def get_linear_system(self):
		"""

		Returns a KShortestCompatibleLinearSystem instance build from the model
		-------

		"""
		return

	def get_enumerator(self):
		"""
		Returns an iterator that yields a single EFM or a list of multiple EFMs of the same size. Call next(iterator) to
		obtain the next set of EFMs.
		"""
		enumerator = self.__algo.get_enumerator(
			linear_system=self.get_linear_system(),
			forced_sets=self.__get_forced_solutions(),
			excluded_sets=self.__get_excluded_solutions())

		for solarg in enumerator:
			self.enumerated_sols.append(solarg)
			yield self.model_reader.decode_k_shortest_solution(solarg)


class KShortestEFMEnumeratorWrapper(KShortestEnumeratorWrapper):
	"""
	Extension of the abstract class KShortestEnumeratorWrapper that takes a metabolic model as input and yields
	elementary flux modes.
	"""

	def __init__(self, model, non_consumed, consumed, produced, **kwargs):
		"""

		Parameters

		----------

			model: A Model instance from the external framework to use. Must be registered in the dict stored as
			external_wrappers.model_readers along with its reader.

			non_consumed: An Iterable[int] or ndarray containing the indices of external metabolites not consumed in the
			model.

			consumed: An Iterable[int] or ndarray containing the indices of external metabolites guaranteed to be produced.

			produced: An Iterable[int] or ndarray containing the indices of external metabolites guaranteed to be consumed.


		"""
		super().__init__(model, **kwargs)
		self.__consumed, self.__non_consumed, self.__produced = consumed, non_consumed, produced

	def get_linear_system(self):
		to_convert = [self.__consumed, self.__non_consumed, self.__produced]
		conv_cn, conv_nc, conv_pr = [[self.model_reader.metabolite_id_to_index(k) for k in lst] for lst in to_convert]
		return IrreversibleLinearSystem(
			S=self.model_reader.S,
			irrev=self.model_reader.irrev_index,
			consumed=conv_cn,
			non_consumed=conv_nc,
			produced=conv_pr)


class KShortestMCSEnumeratorWrapper(KShortestEnumeratorWrapper):
	"""
	Extension of the abstract class KShortestEnumeratorWrapper that takes a metabolic model as input and yields
	minimal cut sets.

	"""

	def __init__(self, model, target_flux_space_dict, target_yield_space_dict, **kwargs):
		super().__init__(model, **kwargs)

		target_flux_space = [tuple([k]) + tuple(v) for k, v in target_flux_space_dict.items()]
		target_yield_space = [k + tuple(v) for k, v in target_yield_space_dict.items()]
		converted_fbs = [DefaultFluxbound.from_tuple(self.model_reader.convert_constraint_ids(t, False)) for t in
						 target_flux_space]
		converted_ybs = [DefaultYieldbound.from_tuple(self.model_reader.convert_constraint_ids(t, True)) for t in
						 target_yield_space]
		self.__ip_constraints = converted_fbs + converted_ybs

	def __materialize_intv_problem(self):
		return InterventionProblem(self.model_reader.S).generate_target_matrix(self.__ip_constraints)

	def get_linear_system(self):
		T, b = self.__materialize_intv_problem()
		return DualLinearSystem(self.model_reader.S, self.model_reader.irrev_index, T, b)

class KShortestEFPEnumeratorWrapper(KShortestEnumeratorWrapper):
	"""
	Extension of the abstract class KShortestEnumeratorWrapper that takes a metabolic model as input and yields
	elementary flux patterns.
	"""

	def __init__(self, model, subset, non_consumed=[], consumed=[], produced=[], **kwargs):
		super().__init__(model, **kwargs)
		self.__subset = subset
		self.__consumed, self.__non_consumed, self.__produced = consumed, non_consumed, produced

	def get_linear_system(self):
		to_convert = [self.__consumed, self.__non_consumed, self.__produced]
		conv_cn, conv_nc, conv_pr = [[self.model_reader.metabolite_id_to_index(k) for k in lst] for lst in to_convert]
		conv_subsets = [self.model_reader.reaction_id_to_index(s) for s in self.__subset]
		return IrreversibleLinearPatternSystem(
			S=self.model_reader.S,
			irrev=self.model_reader.irrev_index,
			subset=conv_subsets,
			consumed=conv_cn,
			non_consumed=conv_nc,
			produced=conv_pr)


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
	def get_model_bounds(self, as_dict):
		"""
		Returns the lower and upper bounds for all fluxes in the model. This either comes in the form of an ordered list
		with tuples of size 2 (lb,ub) or a dictionary with the same tuples mapped by strings with reaction identifiers.

		Parameters

		----------

			as_dict: A boolean value that controls whether the result is a dictionary mapping str to tuple of size 2
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
		return {self.r_ids[k]: v for k, v in sol.attribute_value(sol.SIGNED_VALUE_MAP).items() if v != 0}


class COBRAModelObjectReader(AbstractObjectReader):

	def get_stoichiometric_matrix(self):
		S = np.zeros((len(self.m_ids), len(self.r_ids)))
		for i, r_id in enumerate(self.r_ids):
			for metab, coef in self.model.reactions.get_by_id(r_id).metabolites.items():
				S[self.m_ids.index(metab.id), i] = coef

		return S

	def get_model_bounds(self, as_dict):
		bounds = [r.bounds for r in self.rx_instances]
		if as_dict:
			return dict(zip(self.r_ids, bounds))
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

	def get_model_bounds(self, as_dict):
		bounds = [(r.lb, r.ub) for r in self.rx_instances]
		if as_dict:
			return dict(zip(self.r_ids, bounds))
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

	def get_model_bounds(self, as_dict):
		if as_dict:
			return dict(zip(self.r_ids, self.model.bounds))
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
