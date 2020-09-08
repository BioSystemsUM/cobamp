from cobamp.algorithms.kshortest import *
from cobamp.core.linear_systems import IrreversibleLinearSystem, DualLinearSystem, IrreversibleLinearPatternSystem, \
	GenericDualLinearSystem
from cobamp.wrappers.external_wrappers import model_readers

from itertools import product


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
				 excluded_solutions=None, solver='CPLEX', force_bounds={}, n_threads=0, workmem=None, big_m=False,
				 max_populate_sols_override=None, time_limit=None, big_m_value=None, cut_function=None, extra_args=None,
				 pre_enum_function=None):
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

			force_bounds: A dict mapping reaction indexes (int for now) with tuples containing lower and upper bounds
			An experimental feature meant to force certain phenotypes on EFP/EFMs

			n_threads: An integer value defining the amount of threads available to the solver

			workmem: An integer value defining the amount of memory in MegaBytes available to the solver
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
		self.__algo_properties[K_SHORTEST_MPROPERTY_TYPE_EFP] = self.is_efp
		self.__algo_properties[K_SHORTEST_OPROPERTY_N_THREADS] = n_threads
		self.__algo_properties[K_SHORTEST_OPROPERTY_WORKMEMORY] = workmem
		self.__algo_properties[K_SHORTEST_OPROPERTY_TIMELIMIT] = 0 if time_limit == None else time_limit
		self.__algo_properties[K_SHORTEST_OPROPERTY_BIG_M_CONSTRAINTS] = big_m
		self.__algo_properties[self.__alg_to_prop_name[algorithm_type]] = stop_criteria
		if big_m_value != None:
			self.__algo_properties[K_SHORTEST_OPROPERTY_BIG_M_VALUE] = big_m_value
		if (max_populate_sols_override != None) and algorithm_type == self.ALGORITHM_TYPE_POPULATE:
			self.__algo_properties[K_SHORTEST_OPROPERTY_MAXSOLUTIONS] = max_populate_sols_override


		def preprocess_cuts(cut_list):
			if cut_list is not None:
				new_cuts = []
				for cut in cut_list:
					new_cut = cut
					if not isinstance(cut, KShortestSolution):
						new_cut = [self.model_reader.reaction_id_to_index(k) if isinstance(k, str) else k for k in cut]
					new_cuts.append(new_cut)
				return new_cuts
			else:
				return []

		self.__forced_solutions = preprocess_cuts(forced_solutions)
		self.__excluded_solutions = preprocess_cuts(excluded_solutions)

		self.force_bounds = {self.model_reader.r_ids.index(k): v for k, v in force_bounds.items()}
		self.solver = solver
		self.pre_enum_function = pre_enum_function
		self.cut_function = cut_function
		self.extra_args = extra_args
		self.__setup_algorithm()
		self.enumerated_sols = []

	def __setup_algorithm(self):
		"""
		Creates the algorithms instance

		Returns:

		"""
		self.algo = KShortestEFMAlgorithm(self.__algo_properties, False)

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
		enumerator = self.algo.get_enumerator(
			linear_system=self.get_linear_system(),
			forced_sets=self.__get_forced_solutions(),
			excluded_sets=self.__get_excluded_solutions())

		if self.pre_enum_function is not None:
			self.pre_enum_function(self.algo.ksh, self.extra_args)
		for solarg in enumerator:
			self.enumerated_sols.append(solarg)
			yield self.decode_solution(solarg)
			if self.cut_function is not None:
				cut_arg = solarg if isinstance(solarg, (tuple, list)) else [solarg]
				self.cut_function(cut_arg, self.algo.ksh, self.extra_args)

	def decode_k_shortest_solution(self, sol):
		## TODO: Make MAX_PRECISION a parameter for linear systems or the KShortestAlgorithm
		return {self.model_reader.r_ids[k]: sol.attribute_value(sol.SIGNED_VALUE_MAP)[k]
				for k in sol.get_active_indicator_varids()}

	def decode_solution(self, solarg):
		if isinstance(solarg, (list,tuple)):
			return [self.decode_k_shortest_solution(sol) for sol in solarg]
		else:
			return self.decode_k_shortest_solution(solarg)

class KShortestEFMEnumeratorWrapper(KShortestEnumeratorWrapper):
	"""
	Extension of the abstract class KShortestEnumeratorWrapper that takes a metabolic model as input and yields
	elementary flux modes.
	"""

	def __init__(self, model, non_consumed, consumed, produced, non_produced, subset=None, **kwargs):
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
		self.is_efp = False
		super().__init__(model, **kwargs)
		self.__consumed, self.__non_consumed, self.__produced, self.__non_produced, self.__subset = \
			consumed, non_consumed, produced, non_produced, subset

	def get_linear_system(self):
		to_convert = [self.__consumed, self.__non_consumed, self.__produced, self.__non_produced]
		conv_cn, conv_nc, conv_pr, conv_np = [[self.model_reader.metabolite_id_to_index(k) for k in lst] for lst in to_convert]
		lb, ub = [array(k) for k in self.model_reader.get_model_bounds(as_dict=False, separate_list=True)]
		if self.__subset == None:

			return IrreversibleLinearSystem(
				S=self.model_reader.S,
				lb=lb, ub=ub,
				consumed=conv_cn,
				non_consumed=conv_nc,
				produced=conv_pr,
				non_produced=conv_np,
				solver=self.solver,
				force_bounds=self.force_bounds
			)
		else:
			return IrreversibleLinearPatternSystem(
				S=self.model_reader.S,
				lb=lb, ub=ub,
				consumed=conv_cn,
				non_consumed=conv_nc,
				produced=conv_pr,
				non_produced=conv_np,
				subset=[self.model_reader.reaction_id_to_index(s) for s in self.__subset],
				solver=self.solver,
				force_bounds=self.force_bounds
			)

class KShortestEFPEnumeratorWrapper(KShortestEnumeratorWrapper):
	"""
	Extension of the abstract class KShortestEnumeratorWrapper that takes a metabolic model as input and yields
	elementary flux patterns.
	"""

	def __init__(self, model, subset, non_consumed=[], consumed=[], produced=[], non_produced=[],**kwargs):
		self.is_efp = True
		super().__init__(model, **kwargs)
		self.__subset = subset
		self.__consumed, self.__non_consumed, self.__produced, self.__non_produced = consumed, non_consumed, produced, non_produced

	def get_linear_system(self):
		## TODO:  change irrev to lb/ub structure
		to_convert = [self.__consumed, self.__non_consumed, self.__produced, self.__non_produced]
		lb, ub = [array(k) for k in self.model_reader.get_model_bounds(as_dict=False, separate_list=True)]
		conv_cn, conv_nc, conv_pr, conv_np = [[self.model_reader.metabolite_id_to_index(k) for k in lst] for lst in to_convert]
		conv_subsets = [self.model_reader.reaction_id_to_index(s) for s in self.__subset]
		return IrreversibleLinearPatternSystem(
			S=self.model_reader.S,
			lb=lb, ub=ub,
			subset=conv_subsets,
			consumed=conv_cn,
			non_consumed=conv_nc,
			produced=conv_pr,
			non_produced=conv_np,
			solver=self.solver,
			force_bounds=self.force_bounds
		)




class KShortestMCSEnumeratorWrapper(KShortestEnumeratorWrapper):
	"""
	Extension of the abstract class KShortestEnumeratorWrapper that takes a metabolic model as input and yields
	minimal cut sets.

	"""

	def __init__(self, model, target_flux_space_dict, target_yield_space_dict, **kwargs):
		self.is_efp = False
		super().__init__(model, **kwargs)
		self.__ip_constraints = list(chain(*AbstractConstraint.convert_tuple_intervention_problem(
			target_flux_space_dict, target_yield_space_dict, self.model_reader)))

	def materialize_intv_problem(self):
		return InterventionProblem(self.model_reader.S).generate_target_matrix(self.__ip_constraints)

	def get_linear_system(self):
		lb, ub = [array(k) for k in self.model_reader.get_model_bounds(separate_list=True, as_dict=False)]
		T, b = self.materialize_intv_problem()
		return DualLinearSystem(self.model_reader.S, lb, ub, T, b, solver=self.solver)


class KShortestGenericMCSEnumeratorWrapper(KShortestEnumeratorWrapper):

	def __init__(self, model, target_flux_space_dict, target_yield_space_dict, dual_matrix, dual_var_mapper, **kwargs):
		self.is_efp = False
		super().__init__(model, **kwargs)
		self.__ip_constraints = list(chain(*AbstractConstraint.convert_tuple_intervention_problem(
			target_flux_space_dict, target_yield_space_dict, self.model_reader)))

		self.dual_matrix, self.dual_var_mapper = dual_matrix, {v:k for k,v in dual_var_mapper.items()}

	def decode_k_shortest_solution(self, sol):
		## TODO: Make MAX_PRECISION a parameter for linear systems or the KShortestAlgorithm
		mapper = self.dual_var_mapper if self.dual_var_mapper is not None else self.model_reader.r_ids
		return {mapper[k]: sol.attribute_value(sol.SIGNED_VALUE_MAP)[k] for k in sol.get_active_indicator_varids()}

	def get_linear_system(self):
		T, b = InterventionProblem(self.model_reader.S).generate_target_matrix(self.__ip_constraints)
		return GenericDualLinearSystem(self.model_reader.S, self.dual_matrix, T, b, solver=self.solver)


class KShortestGeneticMCSEnumeratorWrapper(KShortestGenericMCSEnumeratorWrapper):
	@staticmethod
	def gene_cut_function(solx, ksh, extra_args):
		alternative_gene_identity = extra_args['F']

		for sol in solx:
			act_vars = sol.get_active_indicator_varids()
			if len(act_vars) > 1:
				dependencies = list(chain(*product(filter(lambda x: len(x) > 0,
														  [list(alternative_gene_identity[av]) for av in act_vars]))))
			else:
				dependencies = [[k] for k in [alternative_gene_identity[av] for av in act_vars][0]]
			if len(dependencies) > 0:
				ksh.exclude_solutions(dependencies)

	@staticmethod
	def set_gene_weights(ksh, extra_args):
		ksh.set_objective_expression(extra_args['gene_weights'])

	def __init__(self, model, target_flux_space_dict, target_yield_space_dict, G, gene_map, F, gene_weights, **kwargs):
		super().__init__(model, target_flux_space_dict, target_yield_space_dict, G, gene_map,
						 cut_function=self.gene_cut_function, pre_enum_function=self.set_gene_weights,
						 extra_args={'gene_map': gene_map, 'F': F, 'gene_weights':gene_weights}, **kwargs)