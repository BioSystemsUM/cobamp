from metaconvexpy.convex_analysis.efm_enumeration.kshortest_efms import KShortestEFMAlgorithm
from metaconvexpy.linear_systems.linear_systems import IrreversibleLinearSystem, DualLinearSystem
from metaconvexpy.convex_analysis.mcs_enumeration.intervention_problem import *

import metaconvexpy.convex_analysis.efm_enumeration.kshortest_efm_properties as kp

import numpy as np

class KShortestEnumerator(object):
	"""
	An abstract class for methods involving the K-shortest EFM enumeration algorithm
	"""
	ITERATIVE = 'kse_iterative'
	POPULATE = 'kse_populate'

	__alg_to_prop_name = {
		ITERATIVE: kp.K_SHORTEST_OPROPERTY_MAXSOLUTIONS,
		POPULATE:kp.K_SHORTEST_OPROPERTY_MAXSIZE
	}

	def __init__(self, model, algorithm_type, stop_criteria, forced_solutions=None, excluded_solutions=None):
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
		self.__model_reader = COBRAModelObjectReader(model)
		self.__algo_properties = kp.KShortestProperties()
		self.__algo_properties[self.__alg_to_prop_name[algorithm_type]] = stop_criteria
		self.__forced_solutions = forced_solutions
		self.__excluded_solutions = excluded_solutions

	def __setup_algorithm(self):
		'''
		Creates the algorithm instance
		Returns:

		'''
		self.__algo = KShortestEFMAlgorithm(self.__algo_properties, False)

	def __validate(self):
		"""
		Returns: Boolean value (True if parameters are correctly set up)

		"""
		pass

	def __get_linear_system(self):
		"""
		Implemented by subclass
		Returns: A KShortestCompatibleLinearSystem instance

		"""

	def __get_forced_solutions(self):
		"""
		Returns: A KShortestCompatibleLinearSystem instance

		"""

	def __get_excluded_solutions(self):
		"""
		Returns: A KShortestCompatibleLinearSystem instance

		"""

	def __get_enumerator(self):
		self.__algo.get_enumerator(
			linear_system=self.__get_linear_system(),
			forced_sets=self.__get_forced_solutions(),
			excluded_sets=self.__get_excluded_solutions())


class KShortestEFMEnumerator(KShortestEnumerator):

	def __get_linear_system(self, *args, **kwargs):
		system = IrreversibleLinearSystem(S, irrev, non_consumed, consumed, produced)
		return system

class KShortestMCSEnumerator(KShortestEnumerator):

	def __init__(self, *args, target_flux_space, target_yield_space, **kwargs):
		super().__init__(*args, **kwargs)
		converted_fbs = [DefaultFluxbound.from_tuple(self.__model_reader.convert_constraint_ids(t, False)) for t in target_flux_space]
		converted_ybs = [DefaultYieldbound.from_tuple(self.__model_reader.convert_constraint_ids(t, True)) for t in target_yield_space]
		self.__ip_constraints = converted_fbs + converted_ybs

	def __materialize_intv_problem(self):
		return InterventionProblem(self.__model_reader.S).generate_target_matrix(self.__ip_constraints)

	def __get_linear_system(self):
		T, b = self.__materialize_intv_problem()
		system = DualLinearSystem(self.__model_reader.S, self.__model_reader.irrev_bool, T, b)
		return system

class COBRAModelObjectReader(object):
	def __init__(self, model):
		self.__model = model
		self.__r_ids, self.__m_ids = self.get_metabolite_and_reactions_ids()
		self.__rx_instances = [model.reactions.get_by_id(rx) for rx in self.__r_ids]

		self.S = self.get_stoichiometric_matrix()
		self.irrev_bool = self.get_irreversibilities()
		self.irrev_index = self.get_irreversibilities(True)
		self.lb, self.ub = tuple(zip(*self.get_model_bounds()))
		self.bounds_dict = self.get_model_bounds(True)


	def get_stoichiometric_matrix(self):
		S = np.zeros((self.__r_ids, self.__m_ids))
		for i,r_id in enumerate(self.__r_ids):
			for metab, coef in self.__model.reactions.get_by_id(r_id).items()
				S[self.__m_ids.index(metab.id)] = coef

		return S

	def get_model_bounds(self, as_dict):
		bounds = [(r.lb, r.ub) for r in self.__rx_instances]
		if as_dict:
			return dict(zip(self.__r_ids, bounds))
		else:
			return tuple(bounds)

	def get_irreversibilities(self, as_index):
		irrev = [not r.reversible for r in self.__rx_instances]
		if as_index:
			irrev = list(np.where(irrev)[0])
		return irrev

	def get_metabolite_and_reactions_ids(self):
		return tuple([[x.id for x in lst] for lst in (model.reactions, model.metabolites)])

	def reaction_id_to_index(self, id):
		return self.__r_ids.index(id)

	def metabolite_id_to_index(self, id):
		return self.__m_ids.index(id)

	def convert_constraint_ids(self, tup, yield_constaint):
		if yield_constaint:
			constraint = tuple(list(map(self.reaction_id_to_index,tup[:2])) + list(tup[2:]))
		else:
			constraint = tuple([self.reaction_id_to_index(tup[0])] + list(tup[1:]))
		return constraint


model = cobra.io.sbml3.read_sbml_model("/home/skapur/MEOCloud/Projectos/DeYeast/Models/iMM904/iMM904_peroxisome.xml")