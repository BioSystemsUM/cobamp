"""
This module includes classes that implement the K-shortest EFM enumeration algorithms. Please refer to the original
authors' paper describing the method [1]. Additional improvements concerning enumeration of EFMs by size have been
adapted from the methods described by Von Kamp et al. [2]

References:

	[1] De Figueiredo, Luis F., et al. "Computing the shortest elementary flux modes in genome-scale metabolic networks"
	Bioinformatics 25.23 (2009): 3158-3165.
	[2] von Kamp, Axel, and Steffen Klamt. "Enumeration of smallest intervention strategies in genome-scale metabolic
	networks" PLoS computational biology 10.1 (2014): e1003378.

"""
import abc

from itertools import chain
from numpy import concatenate, array, zeros, hstack, ones, identity

from cobamp.core.optimization import LinearSystemOptimizer, KShortestSolution
from cobamp.core.linear_systems import IrreversibleLinearPatternSystem, VAR_BINARY
from ..utilities.property_management import PropertyDictionary
import warnings

decompose_list = lambda a: chain.from_iterable(map(lambda i: i if isinstance(i, list) else [i], a))


def value_map_apply(single_fx, pair_fx, value_map, **kwargs):
	return [
		pair_fx(varlist, value_map, **kwargs) if isinstance(varlist, tuple) else single_fx(varlist, value_map, **kwargs)
		for varlist in value_map.keys()]


K_SHORTEST_MPROPERTY_METHOD = 'METHOD'
K_SHORTEST_METHOD_ITERATE = "ITERATE"
K_SHORTEST_METHOD_POPULATE = "POPULATE"

K_SHORTEST_OPROPERTY_MAXSIZE = 'MAXSIZE'
K_SHORTEST_OPROPERTY_MAXSOLUTIONS = "MAXSOLUTIONS"
K_SHORTEST_BIG_M_VALUE = "BIGMVALUE"

kshortest_mandatory_properties = {
	K_SHORTEST_MPROPERTY_METHOD: [K_SHORTEST_METHOD_ITERATE, K_SHORTEST_METHOD_POPULATE]}

kshortest_optional_properties = {
	K_SHORTEST_OPROPERTY_MAXSIZE: lambda x: x > 0 and isinstance(x, int),
	K_SHORTEST_OPROPERTY_MAXSOLUTIONS: lambda x: x > 0 and isinstance(x, int),
	K_SHORTEST_BIG_M_VALUE: lambda x: isinstance(x, (float, int))
}


class KShortestProperties(PropertyDictionary):
	"""
	Class defining a configuration for the K-shortest algorithms.
	The following fields are mandatory:
	K_SHORTEST_MPROPERTY_METHOD:
		- K_SHORTEST_METHOD_ITERATE : Iterative enumeration (one EFM at a time)
		- K_SHORTEST_METHOD_POPULATE : Enumeration by size (EFMs of a certain size at a time)
	"""

	def __init__(self):
		super().__init__(kshortest_mandatory_properties, kshortest_optional_properties)


class KShortestEnumerator(object):
	"""
	Class implementing the k-shortest elementary flux mode algorithms. This is a lower level class implemented using the
	Cplex solver as base. Maybe in the future, this will be readapted for the optlang wrapper, if performance issues
	do not arise.
	"""

	ENUMERATION_METHOD_ITERATE = 'iterate'
	ENUMERATION_METHOD_POPULATE = 'populate'

	def __init__(self, linear_system, m_value=None):

		"""

		Parameters

		----------

			linear_system: A KShortestCompatibleLinearSystem/<LinearSystem> subclass

		"""

		linear_system.build_problem()
		self.__dvar_mapping = linear_system.get_dvar_mapping()
		self.__ls_shape = linear_system.get_stoich_matrix_shape()
		self.model = linear_system
		self.__dvars = linear_system.get_dvars()
		self.__c = linear_system.get_c_variable()
		self.__solzip = lambda x: zip(self.model._get_variables_names(), x)

		# TODO: Find a way to estimate the best possible value for this
		self.__m_value = 10e6 if m_value == None else m_value

		# Open log files
		# self.resf = open('results', 'w')
		# self.logf = open('log', 'w')

		# Setup CPLEX parameters
		self.__set_model_parameters()
		self.is_efp_problem = isinstance(linear_system, IrreversibleLinearPatternSystem)

		# Setup k-shortest constraints
		self.indicator_map = {}
		self.__ivars = []
		big_m = linear_system.solver != 'CPLEX'
		if big_m:
			warnstr = linear_system.solver + ' does not support indicator constraints. Using Big-M constraints with M= ' + str(
				self.__m_value)
			warnings.warn(warnstr)
			self.__add_kshortest_indicators_big_m()
		else:
			self.__add_kshortest_indicators()

		if not self.is_efp_problem:
			self.__add_exclusivity_constraints()

		self.__size_constraint = None
		self.__efp_auxiliary_map = None

		if self.is_efp_problem:
			self.__add_efp_auxiliary_constraints()

		self.__objective_expression = dict(zip(list(range(len(self.__dvars))), [1] * len(self.__dvars)))

		self.__set_objective()
		self.__integer_cuts = []
		self.__exclusion_cuts = []
		self.set_size_constraint(1)
		self.__current_size = 1
		self.optimizer = LinearSystemOptimizer(self.model, build=False)

	def __set_model_parameters(self):
		parset_func = {'CPLEX': self.__set_model_parameters_cplex,
					   'GUROBI': self.__set_model_parameters_gurobi}

		if self.model.solver in parset_func.keys():
			parset_func[self.model.solver]()

	def __set_model_parameters_cplex(self):

		"""
		Internal method to set model parameters. This is based on the original MATLAB code by Von Kamp et al.

		-------
		"""
		instance = self.model.model.problem

		instance.parameters.mip.tolerances.integrality.set(1e-9)
		instance.parameters.clocktype.set(1)
		instance.parameters.advance.set(0)
		instance.parameters.mip.strategy.fpheur.set(1)
		instance.parameters.emphasis.mip.set(2)
		instance.parameters.mip.limits.populate.set(1000000)
		instance.parameters.mip.pool.intensity.set(4)
		instance.parameters.mip.pool.absgap.set(0)
		instance.parameters.mip.pool.replace.set(2)

	# instance.parameters.mip.tolerances.absmipgap.set(0)

	def __set_model_parameters_gurobi(self):

		instance = self.model.model.problem

		instance.params.PoolGap = 0
		instance.params.MIPFocus = 2
		instance.params.MIPAbsGap = 0
		instance.params.PoolSearchMode = 2

	##TODO: Make this more flexible in the future. 4GB of RAM should be enough but some problems might require more.

	def __add_cuts(self, sols, length_override, equality):

		for sol in sols:
			if isinstance(sol, KShortestSolution):
				self.__add_integer_cut(sol.var_values(), efp_cut=self.is_efp_problem, equality=equality,
									   length_override=length_override)
			elif isinstance(sol, list) or isinstance(sol, tuple):
				values = {i: 0 for i in range(len(self.model.model.variables))}
				for k in sol:
					dvars = self.__dvar_mapping[k]
					if isinstance(dvars, (list, tuple)):
						for v in dvars:
							values[self.indicator_map[self.__dvars[v]]] = 1
					else:
						values[self.indicator_map[self.__dvars[dvars]]] = 1
					values[k] = 1
				self.__add_integer_cut(values, efp_cut=self.is_efp_problem, equality=equality,
									   length_override=length_override)

	def exclude_solutions(self, sols):

		"""
		Excludes the supplied solutions from the search by adding them as integer cuts.
		Use at your own discretion as this will yield different EFMs than would be intended.
		This can also be used to exclude certain reactions from the search by supplying solutions with one reaction.

		Parameters

		----------

			sols: An Iterable containing list/tuples with active reaction combinations to exclude or Solution instances.

		-------

		"""
		self.__add_cuts(sols, length_override=1, equality=False)

	def force_solutions(self, sols):
		"""
		Forces a set of reactions encoded as solutions to appear in the subsequent elementary modes to be calculated.

		Parameters

		----------

			sols: An Iterable containing list/tuples with active reaction combinations to exclude or Solution instances.

		-------

		"""
		self.__add_cuts(sols, length_override=0, equality=True)

	def __add_kshortest_indicators(self, chunksize=2000):
		for i in range(0, len(self.__dvars), chunksize):
			print('Adding chunk:',i,i+chunksize)
			dvl = self.__dvars[i:i+chunksize]
			self.__add_kshortest_indicators_from_dvar(dvl)

	def __add_kshortest_indicators_from_dvar(self, dvars):
		"""
		Adds indicator variable to a copy of the supplied linear problem.
		This uses the __dvars map to obtain a list of all variables and assigns an indicator to them.

		-------

		"""
		ilb, iub = [0] * 5, [1] * 5
		itype = VAR_BINARY
		template_matrix = array(
			[[1, 1, 0, 0, 0], [0, -1, 1, 0, 0], [-1, 0, 0, 1, 0], [0, 0, 0, -1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
		)

		template_blb, template_bub = [1, 0, 0, 0, 0, 0], [1, None, 0, None, 0, None]

		row_blb, row_bub = [t * len(dvars) for t in (template_blb, template_bub)]
		helpers = [[k + r for r in [str(i) for i in dvars]] for k in ['i', 'a', 'b', 'c', 'd']]

		vlist = []
		offset = len(self.model.model.variables)

		for vars in zip(*helpers):
			vlist.extend(self.model.add_variables_to_model(vars, lb=ilb, ub=iub, var_types=itype))
		trows, tcols = template_matrix.shape
		template_full = zeros((trows * len(dvars), tcols * len(dvars)))
		diag = zeros((trows * len(dvars), len(dvars)))
		crow = zeros((trows * len(dvars),))
		indicators = []
		for i in range(len(dvars)):
			template_full[i * trows: (i + 1) * trows, i * tcols:(i + 1) * tcols] = template_matrix
			diag[(i * trows) + 4][i] = 1
			diag[(i * trows) + 5][i] = 1
			crow[(i * trows) + 5] = -1
			indicators.append(tuple([(i * trows) + 4, len(dvars) + (i * tcols) + 2, 1]))
			indicators.append(tuple([(i * trows) + 5, len(dvars) + (i * tcols) + 4, 1]))

		nrowmat = hstack([diag, template_full, crow.reshape(-1, 1)])
		vlist += [self.model.get_c_variable()]
		vlist = [self.model.model.variables[i] for i in dvars] + vlist
		new_ivars = [(i * 5) + offset for i in range(len(dvars))]
		self.__ivars.extend(new_ivars)
		self.model.add_rows_to_model(nrowmat, row_blb, row_bub, only_nonzero=True, indicator_rows=indicators,
									 vars=vlist)
		self.indicator_map.update(dict(zip(dvars, new_ivars)))

	def __add_kshortest_indicators_big_m(self):
		dvars = self.__dvars

		M = self.__m_value
		template_matrix = array([[1, -M], [-1, 1]])

		nrowmat = zeros([len(dvars) * 2] * 2)
		for i in range(len(dvars)):
			nrowmat[i * 2:i * 2 + 2, i * 2:i * 2 + 2] = template_matrix

		offset = len(self.model.model.variables)

		ivar_instances = self.model.add_variables_to_model(['i' + str(i) for i in range(len(dvars))],
														   lb=[0] * len(dvars), ub=[1] * len(dvars),
														   var_types=VAR_BINARY)
		dvar_instances = [self.model.model.variables[i] for i in dvars]
		vlist = list(chain(*list(zip(dvar_instances, ivar_instances))))
		self.__ivars = [i + offset for i in range(len(dvars))]

		self.model.add_rows_to_model(nrowmat, [None] * (len(dvars) * 2), [0] * (len(dvars) * 2), only_nonzero=True,
									 vars=vlist)
		self.indicator_map = dict(zip(dvars, self.__ivars))

	def __add_efp_auxiliary_constraints(self):
		self.__efp_auxiliary_map = {}
		itype = VAR_BINARY
		indicator_vars = [self.model.model.variables[v] for v in self.__ivars]
		ilb, iub = [0] * len(indicator_vars), [1] * len(indicator_vars)
		offset = len(self.model.model.variables)
		helpers = self.model.add_variables_to_model(['efp_h' + str(i) for i in range(len(indicator_vars))], lb=ilb,
													ub=iub, var_types=itype)

		self.__efp_auxiliary_map = dict(zip(self.__ivars, [offset+i for i in range(len(helpers))]))
		vlist = indicator_vars + helpers
		mat_template = identity(len(self.indicator_map))
		mat = hstack([mat_template, -mat_template])
		rhs_l = [0] * len(self.indicator_map)
		rhs_u = [None] * len(self.indicator_map)

		print('MILP2')
		## Adding MILP2
		self.model.add_rows_to_model(mat, rhs_l, rhs_u, only_nonzero=False, indicator_rows=None, vars=vlist,
									 names=None)

		print('MILP4')
		## Adding MILP4
		self.model.add_rows_to_model(ones([1, len(indicator_vars)]), [1], [None], only_nonzero=False,
									 indicator_rows=None, vars=helpers, names=None)

	def __add_exclusivity_constraints(self):
		"""
		Adds constraints so that fluxes with two assigned dvars will only have one of the indicators active (flux must
		not be carried through both senses at once to avoid cancellation)

		-------

		"""
		exclusive_dvars = {k: v for k, v in self.__dvar_mapping.items() if isinstance(v, (tuple, list))}
		M, N = len(exclusive_dvars), len(self.__dvars)
		smat = zeros((M, N))
		for k, n in enumerate(exclusive_dvars.items()):
			idx, v = n
			if not isinstance(v, (tuple, list)):
				v = array([v])
			else:
				v = array(v)
			smat[k][v] = 1

		self.model.add_rows_to_model(smat, [None] * M, [1] * M, True,
									 vars=[self.model.model.variables[self.indicator_map[k]] for k in self.__dvars])

	def __set_objective(self):
		"""
		Defines the objective for the optimization problem (Minimize the sum of all indicator variables)

		-------

		"""
		vars = [self.model.model.variables[i] for i in self.__ivars]
		self.model.set_objective(ones(len(vars), ), minimize=True, vars=vars)

	def __integer_cut_count(self):
		"""

		Returns the amount of integer cuts added to the model

		-------

		"""

		return len(self.__integer_cuts)

	def __add_integer_cut(self, value_map, efp_cut=False, equality=False, length_override=1, eps=1e-6):
		"""
		Adds an integer cut based on a map of flux values (from a solution).

		Parameters

		----------

			value_map: A dictionary mapping solver variables with values

			force_sol: Boolean value indicating whether the solution is to be excluded or forced.

		-------

		"""
		if efp_cut:
			assert self.__efp_auxiliary_map is not None, 'Error: trying to set an integer cut for an EFP problem without any auxiliary variable'

		cut_length, cut_vars = 0, []

		for var, dvl in self.__dvar_mapping.items():
			if not isinstance(dvl, (tuple, list)):
				dvl = [dvl]
			dvars_idx = [self.__dvars[k] for k in dvl]
			indicator_idx = [self.indicator_map[k] for k in dvars_idx]
			if sum([abs(value_map[i]) for i in indicator_idx]) > eps:
				cut_vars.extend([self.model.model.variables[k] for k in indicator_idx])
				if efp_cut:
					cut_vars.extend([self.model.model.variables[self.__efp_auxiliary_map[k]] for k in indicator_idx])
				cut_length += 1

		rhs_value = cut_length - (length_override * int(not efp_cut))
		cut = self.model.add_rows_to_model(
			S_new=ones((1, len(cut_vars))),
			b_lb=[rhs_value if equality else None],
			b_ub=[rhs_value],
			only_nonzero=True,
			vars=cut_vars
		)
		cut[0].name = '_'.join(
			[str(k) for k in ['IC_OV', length_override, 'EQ' if equality else 'LE', len(self.__integer_cuts)]])
		self.__integer_cuts.append(cut)

	def set_size_constraint(self, start_at, equal=False):
		"""
		Defines the size constraint for the K-shortest algorithms.

		Parameters

		----------

			start_at: Size from which the solutions will be obtained.

			equal: Boolean indicating whether the solutions will match the size or can be higher than it.

		-------

		"""
		# TODO: Find a way to add a single constraint with two bounds.
		self.model.model.update()
		if 'KSH_SizeConstraint_' in self.model.model.constraints:
			cns = self.model.model.constraints['KSH_SizeConstraint_']
			self.model.set_constraint_bounds([cns], [start_at], [start_at if equal else None])
		else:
			c = ones((1, len(self.__ivars)))
			vars = [self.model.model.variables[i] for i in self.__ivars]
			constraint = \
			self.model.add_rows_to_model(c, [start_at], [start_at if equal else None], only_nonzero=False, vars=vars,
										 names=['KSH_SizeConstraint_'])[0]
			self.model.model.update()

	def get_model(self):
		"""

		Returns the solver instance.

		-------

		"""
		return self.model

	def __optimize(self):
		"""

		Optimizes the model and returns a single KShortestSolution instance for the model, adding an exclusion integer
		cut for it.

		-------

		"""
		try:
			sol = self.optimizer.optimize()
			status = sol.status()
			if status == 'optimal':
				var_values = dict(zip(list(range(len(sol.x()))), sol.x()))
				sol = KShortestSolution(var_values, status, self.indicator_map, self.__dvar_mapping, self.__dvars)
				return sol
		except Exception as e:
			print(e)

	def __populate(self):
		"""
		Finds all feasible MIP solutions for the problem and returns them as a list of KShortestSolution instances.

		Returns a list of KShortestSolution instances

		-------

		"""
		# self.model.write('indicator_efmmodel.lp') ## For debug purposes
		sols = []
		try:
			rawsols = self.optimizer.populate(999999)
			for sol in rawsols:
				var_values = dict(zip(list(range(len(sol.x()))), sol.x()))
				sols.append(KShortestSolution(var_values, None, self.indicator_map, self.__dvar_mapping, self.__dvars))
			for sol in sols:
				self.__add_integer_cut(sol.var_values(), efp_cut=self.is_efp_problem)
			return sols
		except Exception as e:
			raise e

	def solution_iterator(self, maximum_amount=2 ** 31 - 1):
		"""
		Generates a solution iterator. Each next call will yield a single solution. This method should be used to allow
		flexibility when enumerating EFMs for large problems. Since it uses the optimize routine, this may be slower in
		the longer term.

		-------

		"""
		i = 0
		self.set_size_constraint(1)
		failed = False
		while not failed and i < maximum_amount:
			try:
				result = self.get_single_solution()
				i += 1
				yield result
			except Exception as e:
				print('Enumeration ended:', e)
				failed = True

	def population_iterator(self, max_size):
		"""
		Generates a solution iterator that yields a list of solutions. Each next call returns all EFMs for a single size
		starting from 1 up to max_size.

		Parameters

		----------

			max_size: The maximum solution size.


			Returns a list of KShortestSolution instances.

		-------


		"""
		for i in range(1, max_size + 1):
			# print('Starting size', str(i))
			try:
				self.set_size_constraint(i, True)
				sols = self.populate_current_size()
				yield sols if sols is not None else []
			except Exception as e:
				print('No solutions or error occurred at size ', i)
				raise e

	def populate_current_size(self):
		"""

		Returns the solutions for the current size. Use the population_iterator method instead.

		-------

		"""
		sols = self.__populate()
		return sols

	def get_single_solution(self):
		"""

		Returns a single solution. Use the solution_iterator method instead.

		-------

		"""
		sol = self.__optimize()
		if sol is None:
			raise Exception('Solution is empty')
		self.__add_integer_cut(sol.var_values(), efp_cut=self.is_efp_problem)
		return sol

	def reset_enumerator_state(self):
		"""

		Resets all integer cuts and size constraints.

		-------

		"""
		self.optimizer = LinearSystemOptimizer(self.model, build=False)
		self.model.model.remove(self.__integer_cuts)
		self.__integer_cuts = []
		self.set_size_constraint(1)


class KShortestEFMAlgorithm(object):
	"""
	A higher level class to use the K-Shortest algorithms. This encompasses the standard routine for enumeration of EFMs.
	Requires a configuration defining an algorithms type. See <KShortestProperties>
	"""

	def __init__(self, configuration, verbose=True):
		"""

		Parameters

		----------

			configuration: A KShortestProperties instance

			verbose: Boolean indicating whether to print useful messages while enumerating.

		"""
		assert configuration.__class__ == KShortestProperties, 'Configuration class is not KShortestProperties'
		self.configuration = configuration
		self.verbose = verbose

	def __prepare(self, linear_system, excluded_sets, forced_sets):
		## TODO: Change this method's name
		"""
		Enumerates the elementary modes for a linear system

		Parameters

		----------

			linear_system: A KShortestCompatibleLinearSystem instance

			excluded_sets: Iterable[Tuple[Solution/Tuple]] with solutions to exclude from the enumeration

			forced_sets: Iterable[Tuple[Solution/Tuple]] with solutions to force

		-------
		Returns a list with solutions encoding elementary flux modes.

		"""
		assert self.configuration.has_required_properties(), "Algorithm configuration is missing required parameters."
		self.ksh = KShortestEnumerator(linear_system, m_value=self.configuration[K_SHORTEST_BIG_M_VALUE])
		if excluded_sets is not None:
			self.ksh.exclude_solutions(excluded_sets)
		if forced_sets is not None:
			self.ksh.force_solutions(forced_sets)

	def enumerate(self, linear_system, excluded_sets=None, forced_sets=None):
		"""
		Enumerates the elementary modes for a linear system

		Parameters

		----------

			linear_system: A KShortestCompatibleLinearSystem instance

			excluded_sets: Iterable[Tuple[Solution/Tuple]] with solutions to exclude from the enumeration

			forced_sets: Iterable[Tuple[Solution/Tuple]] with solutions to force

		-------

		Returns a list with solutions encoding elementary flux modes.

		"""
		enumerator = self.get_enumerator(linear_system, excluded_sets, forced_sets)
		sols = list(enumerator)
		if self.configuration[K_SHORTEST_MPROPERTY_METHOD] == K_SHORTEST_METHOD_POPULATE:
			sols = list(chain(*sols))
		# DEBUG
		# linear_system.write_to_lp('test.lp')
		return sols

	def get_enumerator(self, linear_system, excluded_sets, forced_sets):
		"""


		Parameters

		----------

			linear_system: A KShortestCompatibleLinearSystem instance

			excluded_sets: Iterable[Tuple[Solution/Tuple]] with solutions to exclude from the enumeration

			forced_sets: Iterable[Tuple[Solution/Tuple]] with solutions to force

		Returns an iterator that yields one or multiple EFMs at each iteration, depending on the properties.

		"""
		self.__prepare(linear_system, excluded_sets, forced_sets)

		if self.configuration[K_SHORTEST_MPROPERTY_METHOD] == K_SHORTEST_METHOD_ITERATE:
			limit = self.configuration[K_SHORTEST_OPROPERTY_MAXSOLUTIONS]
			if limit is None:
				limit = 1
				warnings.warn(Warning(
					'You have not defined a maximum solution size for the enumeration process. Defaulting to 1.'))
			return self.ksh.solution_iterator(limit)

		elif self.configuration[K_SHORTEST_MPROPERTY_METHOD] == K_SHORTEST_METHOD_POPULATE:
			limit = self.configuration[K_SHORTEST_OPROPERTY_MAXSIZE]
			if limit is None:
				warnings.warn(
					Warning('You have not defined a maximum size for the enumeration process. Defaulting to size 1.'))
				limit = 1
			return self.ksh.population_iterator(limit)
		else:
			raise Exception('Algorithm type is invalid! If you see this message, something went wrong!')


### Parameters


class InterventionProblem(object):
	"""
	Class containing functions useful when defining an problem using the intervention problem framework.
	References:
		[1] Hädicke, O., & Klamt, S. (2011). Computing complex metabolic intervention strategies using constrained
		minimal cut sets. Metabolic engineering, 13(2), 204-213.
	"""

	def __init__(self, S):
		"""
		Object that generates target matrices for a given set of constraints admissible for an intervention problem

		Parameters
		----------

			S: The stoichiometric matrix used to generate the enumeration problem
		"""
		self.__num_rx = S.shape[1]

	def generate_target_matrix(self, constraints):
		"""

		Parameters:
		----------
			constraints: An iterable containing valid constraints of

		Returns a tuple (T,b) with two elements:
			T is numpy 2D array with as many rows specifying individual bounds (lower and upper bounds count as two) for
			each reaction.

			b is a numpy 1D array with the right hand side of the T.v > b inequality. This represents the value of the
			bound.

		"""
		constraint_pairs = [const.materialize(self.__num_rx) for const in constraints]
		Tlist, blist = list(zip(*constraint_pairs))

		T = concatenate(Tlist, axis=0)
		b = array(list(chain(*blist)))
		return T, b


class AbstractConstraint(object):
	"""
	Object representing a constraint to be used within the intervention problem structures provided in this package.
	"""
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def materialize(self, n):
		"""
		Generates a matrix T 1-by-n or 2-by-n and a list b of length 1 or 2 to be used for target flux vector
		definition within the intervention problem framework

		Parameters:

			n: Number of columns to include in the target matrix

		Returns: Tuple with Iterable[ndarray] and list of float/int

		"""
		return

	@abc.abstractmethod
	def from_tuple(tup):
		"""
		Generates a constraint from a tuple. Refer to subclasses for each specific format.

		Returns
		-------

		"""
		return


class DefaultFluxbound(AbstractConstraint):
	"""
	Class representing bounds for a single flux with a lower and an upper bound.
	"""

	def __init__(self, lb, ub, r_index):
		"""
		Parameters
		----------
			lb: Numerical lower bound

			ub: Numerical upper bound

			r_index: Reaction index on the stoichiometric matrix to which this bound belongs
		"""

		self.__r_index = r_index
		self.__lb = lb
		self.__ub = ub

	def materialize(self, n):
		Tx = []
		b = []
		if self.__lb != None:
			Tlb = zeros((1, n))
			Tlb[0, self.__r_index] = -1
			b.append(-self.__lb)
			Tx.append(Tlb)
		if self.__ub != None:
			Tub = zeros((1, n))
			Tub[0, self.__r_index] = 1
			b.append(self.__ub)
			Tx.append(Tub)

		return concatenate(Tx, axis=0), b

	def from_tuple(tup):
		"""

		Returns a DefaultFluxbound instance from a tuple containing a reaction index as well as lower and upper bounds.
		-------

		"""
		index, lb, ub = tup
		return DefaultFluxbound(lb, ub, index)


class DefaultYieldbound(AbstractConstraint):
	"""
	Class representing a constraint on a yield between two fluxes. Formally, this constraint can be represented as
	n - yd < b, assuming n and d as the numerator and denominator fluxes (yield(N,D) = N/D), y as the maximum yield and
	b as a deviation value.
	"""

	def __init__(self, lb, ub, numerator_index, denominator_index, deviation=0):
		"""

		Parameters
		----------
			lb: numerical lower bound

			ub: numerical upper bound

			numerator_index: reaction index for the flux in the numerator

			denominator_index: reaction index for the flux in the denominator

			deviation: numerical deviation for the target space
		"""
		self.__lb = lb
		self.__ub = ub
		self.__numerator_index = numerator_index
		self.__denominator_index = denominator_index
		self.__deviation = deviation if not deviation is None else 0

	def materialize(self, n):
		Tx = []
		b = []
		if self.__lb != None:
			Tlb = zeros((1, n))
			Tlb[0, self.__numerator_index] = -1
			Tlb[0, self.__denominator_index] = self.__lb
			b.append(self.__deviation)
			Tx.append(Tlb)
		if self.__ub != None:
			Tub = zeros((1, n))
			Tub[0, self.__numerator_index] = 1
			Tub[0, self.__denominator_index] = - self.__ub
			b.append(self.__deviation)
			Tx.append(Tub)

		return concatenate(Tx, axis=0), b

	def from_tuple(tup):
		"""

		Returns a DefaultYieldbound instance from a tuple containing numerator and denominator indices, yield lower and
		upper bounds, a flag indicating whether it's an upper bound and a deviation (optional)
		-------

		"""
		n, d, ylb, yub = tup[:4]
		if len(tup) > 4:
			dev = tup[4]

		return DefaultYieldbound(ylb, yub, n, d, dev)
