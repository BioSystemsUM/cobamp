import abc

from cobamp.algorithms.kshortest import *
from cobamp.core.linear_systems import IrreversibleLinearSystem, DualLinearSystem, IrreversibleLinearPatternSystem
from cobamp.wrappers.external_wrappers import model_readers


class KShortestEnumeratorWrapper(object):
	__metaclass__ = abc.ABCMeta
	"""
	An abstract class for methods involving the K-shortest EFM enumeration algorithms
	"""
	ALGORITHM_TYPE_ITERATIVE = 'kse_iterative'
	ALGORITHM_TYPE_POPULATE = 'kse_populate'

	__alg_to_prop_name = {
		ALGORITHM_TYPE_ITERATIVE: K_SHORTEST_OPROPERTY_MAXSOLUTIONS,
		ALGORITHM_TYPE_POPULATE: K_SHORTEST_OPROPERTY_MAXSIZE
	}

	__alg_to_alg_name = {
		ALGORITHM_TYPE_ITERATIVE: K_SHORTEST_METHOD_ITERATE,
		ALGORITHM_TYPE_POPULATE: K_SHORTEST_METHOD_POPULATE
	}

	def __init__(self, model, algorithm_type=ALGORITHM_TYPE_POPULATE, stop_criteria=1, forced_solutions=None,
				 excluded_solutions=None, solver='CPLEX'):
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
		if type(model).__module__ in model_readers.keys():
			self.model_reader = model_readers[type(model).__module__](model)
		else:
			raise TypeError(
				"The `model` instance is not currently supported by cobamp. Currently available readers are: " + str(
					list(model_readers.keys())))

		self.__algo_properties = KShortestProperties()
		self.__algo_properties[K_SHORTEST_MPROPERTY_METHOD] = self.__alg_to_alg_name[algorithm_type]
		self.__algo_properties[self.__alg_to_prop_name[algorithm_type]] = stop_criteria
		self.__forced_solutions = forced_solutions
		self.__excluded_solutions = excluded_solutions
		self.solver = solver
		self.__setup_algorithm()
		self.enumerated_sols = []

	def __setup_algorithm(self):
		"""
		Creates the algorithms instance

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
			produced=conv_pr,
			solver=self.solver)


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
		return DualLinearSystem(self.model_reader.S, self.model_reader.irrev_index, T, b, solver=self.solver)


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
			produced=conv_pr,
			solver=self.solver)
