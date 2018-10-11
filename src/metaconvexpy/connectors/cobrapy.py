from metaconvexpy.convex_analysis.efm_enumeration.kshortest_efms import KShortestEFMAlgorithm
from metaconvexpy.linear_systems.linear_systems import IrreversibleLinearSystem, DualLinearSystem
from metaconvexpy.convex_analysis.mcs_enumeration.intervention_problem import *

import metaconvexpy.convex_analysis.efm_enumeration.kshortest_efm_properties as kp

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
		Initializes the enumerator algorithm with the
		Args:
			model: A Model object
		"""

		self.__model = model
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

	def __get_stoich_matrix(self):
		"""

		Returns: A numpy array with the stoichiometric matrix.

		"""

	def __get_enumerator(self):
		self.__algo.get_enumerator(
			linear_system=self.__get_linear_system(),
			forced_sets=self.__get_forced_solutions(),
			excluded_sets=self.__get_excluded_solutions())


class KShortestEFMEnumerator(KShortestEnumerator):

	def __get_linear_system(self):
		system = IrreversibleLinearSystem(S, irrev, non_consumed, consumed, produced)
		return system

class KShortestMCSEnumerator(KShortestEnumerator):

	def __init__(self, *args, intervention_problem, **kwargs):
		super().__init__(*args, **kwargs)
		self.__intervention_problem = intervention_problem


	def __materialize_intv_problem(self):
		ip = InterventionProblem()

	def __get_linear_system(self):
		T, b = self.__materialize_intv_problem()
		system = DualLinearSystem(S, irrev, T, b)

class COBRAModelObjectReader(object):
	def __init__(self, model):
		self.__model = model

	def get_stoichiometric_matrix(self):

