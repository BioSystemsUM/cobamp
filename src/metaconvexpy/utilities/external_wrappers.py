from metaconvexpy.convex_analysis.efm_enumeration.kshortest_efms import KShortestEFMAlgorithm
from metaconvexpy.linear_systems.linear_systems import IrreversibleLinearSystem, DualLinearSystem
from metaconvexpy.convex_analysis.mcs_enumeration.intervention_problem import *

import metaconvexpy.convex_analysis.efm_enumeration.kshortest_efm_properties as kp

import abc
import numpy as np


class KShortestEnumeratorWrapper(object):
	"""
	An abstract class for methods involving the K-shortest EFM enumeration algorithm
	"""
	ALGORITHM_TYPE_ITERATIVE = 'kse_iterative'
	ALGORITHM_TYPE_POPULATE = 'kse_populate'

	__alg_to_prop_name = {
		ALGORITHM_TYPE_ITERATIVE: kp.K_SHORTEST_OPROPERTY_MAXSOLUTIONS,
		ALGORITHM_TYPE_POPULATE:kp.K_SHORTEST_OPROPERTY_MAXSIZE
	}

	__alg_to_alg_name = {
		ALGORITHM_TYPE_ITERATIVE: kp.K_SHORTEST_METHOD_ITERATE,
		ALGORITHM_TYPE_POPULATE: kp.K_SHORTEST_METHOD_POPULATE
	}

	def __init__(self, model, algorithm_type=ALGORITHM_TYPE_POPULATE, stop_criteria=1, forced_solutions=None, excluded_solutions=None):
		"""

		Parameters
		----------
		model
		algorithm_type
		stop_criteria
		forced_solutions
		excluded_solutions
		"""

		self.__model = model
		if model.__module__ in model_readers.keys():
			self.model_reader = model_readers[model.__module__](model)
		else:
			raise TypeError("The `model` instance is not currently supported by metaconvexpy. Currently available readers are: "+str(list(model_readers.keys())))

		self.__algo_properties = kp.KShortestProperties()
		self.__algo_properties[kp.K_SHORTEST_MPROPERTY_METHOD] = self.__alg_to_alg_name[algorithm_type]
		self.__algo_properties[self.__alg_to_prop_name[algorithm_type]] = stop_criteria
		self.__forced_solutions = forced_solutions
		self.__excluded_solutions = excluded_solutions
		self.__setup_algorithm()

	def __setup_algorithm(self):
		'''
		Creates the algorithm instance
		Returns:

		'''
		self.__algo = KShortestEFMAlgorithm(self.__algo_properties, False)

	def __get_forced_solutions(self):
		"""
		Returns: A KShortestCompatibleLinearSystem instance

		"""
		return self.__forced_solutions

	def __get_excluded_solutions(self):
		"""
		Returns: A KShortestCompatibleLinearSystem instance

		"""
		return self.__excluded_solutions

	def get_linear_system(self):
		"""

		Returns
		-------

		"""
		return

	def get_enumerator(self):
		enumerator = self.__algo.get_enumerator(
			linear_system=self.get_linear_system(),
			forced_sets=self.__get_forced_solutions(),
			excluded_sets=self.__get_excluded_solutions())

		for solarg in enumerator:
			yield self.model_reader.decode_k_shortest_solution(solarg)



class KShortestEFMEnumeratorWrapper(KShortestEnumeratorWrapper):

	def __get_linear_system(self, non_consumed, consumed, produced):
		to_convert = [consumed, non_consumed, produced]
		conv_cn, conv_nc, conv_pr = [[self.model_reader.metabolite_id_to_index(k) for k in lst] for lst in to_convert]
		return IrreversibleLinearSystem(S, irrev, conv_cn, conv_nc, conv_pr)


class KShortestMCSEnumeratorWrapper(KShortestEnumeratorWrapper):

	def __init__(self, model, target_flux_space_dict, target_yield_space_dict, **kwargs):
		super().__init__(model, **kwargs)


		target_flux_space = [tuple([k]) + tuple(v) for k,v in target_flux_space_dict.items()]
		target_yield_space = [k + tuple(v) for k,v in target_yield_space_dict.items()]
		converted_fbs = [DefaultFluxbound.from_tuple(self.model_reader.convert_constraint_ids(t, False)) for t in target_flux_space]
		converted_ybs = [DefaultYieldbound.from_tuple(self.model_reader.convert_constraint_ids(t, True)) for t in target_yield_space]
		self.__ip_constraints = converted_fbs + converted_ybs

	def __materialize_intv_problem(self):
		return InterventionProblem(self.model_reader.S).generate_target_matrix(self.__ip_constraints)

	def get_linear_system(self):
		T, b = self.__materialize_intv_problem()
		return DualLinearSystem(self.model_reader.S, self.model_reader.irrev_index, T, b)

class AbstractObjectReader(object):
	__metaclass__ = abc.ABCMeta

	def __init__(self, model):
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
		return

	@abc.abstractmethod
	def get_model_bounds(self, as_dict):
		return

	@abc.abstractmethod
	def get_irreversibilities(self, as_index):
		return

	@abc.abstractmethod
	def get_rx_instances(self):
		return

	@abc.abstractmethod
	def get_metabolite_and_reactions_ids(self):
		return

	def reaction_id_to_index(self, id):
		return self.r_ids.index(id)

	def metabolite_id_to_index(self, id):
		return self.m_ids.index(id)




	def convert_constraint_ids(self, tup, yield_constraint):
		if yield_constraint:
			constraint = tuple(list(map(self.reaction_id_to_index,tup[:2])) + list(tup[2:]))
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
		for i,r_id in enumerate(self.r_ids):
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


model_readers = {
	'cobra.core.model': COBRAModelObjectReader,
	'framed.model.cbmodel': FramedModelObjectReader
}