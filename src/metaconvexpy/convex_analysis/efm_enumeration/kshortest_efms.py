import cplex
from itertools import chain
from metaconvexpy.linear_systems.optimization import Solution, copy_cplex_model
from metaconvexpy.utilities.property_management import PropertyDictionary
import metaconvexpy.convex_analysis.efm_enumeration.kshortest_efm_properties as kp
import warnings

CPLEX_INFINITY = cplex.infinity
decompose_list = lambda a: chain.from_iterable(map(lambda i: i if isinstance(i, list) else [i], a))


def value_map_apply(single_fx, pair_fx, value_map, **kwargs):
	return [
		pair_fx(varlist, value_map, **kwargs) if isinstance(varlist, tuple) else single_fx(varlist, value_map, **kwargs)
		for varlist in value_map.keys()]


class KShortestEnumerator(object):
	'''
	Class implementing the k-shortest elementary flux mode algorithm. This is a lower level class implemented using the
	Cplex solver as base. Maybe in the future, this will be readapted for the optlang wrapper, if performance issues
	do not arise.
	'''
	ENUMERATION_METHOD_ITERATE = 'iterate'
	ENUMERATION_METHOD_POPULATE = 'populate'
	SIZE_CONSTRAINT_NAME = 'KShortestSizeConstraint'

	def __init__(self, linear_system):
		'''

		Parameters
		----------
		linear_system: A <KShortestCompatibleLinearSystem>/<LinearSystem> subclass
		'''

		# Get linear system constraints and variables
		linear_system.build_problem()
		self.__dvar_mapping = linear_system.get_dvar_mapping()
		self.__ls_shape = linear_system.get_stoich_matrix_shape()
		self.model = copy_cplex_model(linear_system.get_model())
		self.__dvars = linear_system.get_dvars()
		self.__c = linear_system.get_c_variable()
		self.__solzip = lambda x: zip(self.model.variables.get_names(), x)

		# Open log files
		# self.resf = open('results', 'w')
		# self.logf = open('log', 'w')

		# Setup CPLEX parameters
		self.__set_model_parameters()

		# Setup k-shortest constraints
		self.__add_kshortest_indicators()
		self.__add_exclusivity_constraints()
		self.__size_constraint = None
		# TODO: change this to cplex notation
		self.__objective_expression = list(
			zip(list(self.__indicator_map.values()), [1] * len(self.__indicator_map.keys())))
		self.__set_objective()
		self.__integer_cuts = []
		self.__exclusion_cuts = []
		self.set_size_constraint(1)
		self.__current_size = 1

	def __set_model_parameters(self):
		'''
		Internal method to set model parameters. This is based on the original MATLAB code by Von Kamp et al.
		##TODO: Make this more flexible in the future. 4GB of RAM should be enough but some problems might require more.
		-------
		'''
		self.model.parameters.mip.tolerances.integrality.set(1e-9)
		self.model.parameters.workmem.set(4096)
		self.model.parameters.clocktype.set(1)
		self.model.parameters.advance.set(0)
		self.model.parameters.mip.strategy.fpheur.set(1)
		self.model.parameters.emphasis.mip.set(2)
		self.model.set_results_stream(None)
		self.model.set_log_stream(None)
		self.model.parameters.mip.limits.populate.set(1000000)
		self.model.parameters.mip.pool.capacity.set(1000000)
		self.model.parameters.mip.pool.intensity.set(4)
		self.model.parameters.mip.pool.absgap.set(0)
		self.model.parameters.mip.pool.replace.set(2)

	def exclude_solutions(self, sols):
		'''
		Excludes the supplied solutions from the search by adding them as integer cuts.
		Use at your own discretion as this will yield different EFMs than would be intended.
		This can also be used to exclude certain reactions from the search by supplying solutions with one reaction.

		Parameters
		----------
		sols: An Iterable containing list/tuples with active reaction combinations to exclude or Solution instances.
		-------

		'''
		for sol in sols:
			if isinstance(sol, KShortestSolution):
				self.__add_integer_cut(sol.var_values())
			elif isinstance(sol, list) or isinstance(sol, tuple):
				ivars = [self.__indicator_map[k] for k in list(chain(*[self.__dvars[i] for i in sol]))]
				lin_expr = (ivars, [1] * len(ivars))
				sense = ['L']
				rhs = [len(sol) - 1]
				names = ['exclusion_cuts' + str(len(self.__exclusion_cuts))]
				self.model.linear_constraints.add(lin_expr=[lin_expr], senses=sense, rhs=rhs, names=names)

	def force_solutions(self, sols):
		'''
		Forces a set of reactions encoded as solutions to appear in the subsequent elementary modes to be calculated.
		Parameters
		----------
		sols: An Iterable containing list/tuples with active reaction combinations to exclude or Solution instances.
		-------

		'''
		for sol in sols:
			if isinstance(sol, KShortestSolution):
				self.__add_integer_cut(sol.var_values(), force_sol=True)
			elif isinstance(sol, list) or isinstance(sol, tuple):
				ivars = [self.__indicator_map[k] for k in list(chain(*[self.__dvars[i] for i in sol]))]
				lin_expr = (ivars, [1] * len(ivars))
				sense = ['E']
				rhs = [len(sol)]
				names = ['forced_cuts' + str(len(self.__exclusion_cuts))]
				self.model.linear_constraints.add(lin_expr=[lin_expr], senses=sense, rhs=rhs, names=names)


	def __add_kshortest_indicators(self):
		'''
		Adds indicator variable to a copy of the supplied linear problem.
		This uses the __dvars map to obtain a list of all variables and assigns an indicator to them.
		-------

		'''
		btype = self.model.variables.type.binary
		ivars = [[(subvar + "_ind", 0, 1, btype) for subvar in var] if isinstance(var, tuple) else (
			var + "_ind", 0, 1, btype) for var in self.__dvars]

		dvnames = []
		for elem in self.__dvars:
			if type(elem) is not str:
				dvnames = dvnames + list(elem)
			else:
				dvnames.append(elem)

		ivchain = list(decompose_list(ivars))

		ivnames, ivlb, ivub, ivtype = list(zip(*ivchain))
		self.model.variables.add(names=ivnames, lb=ivlb, ub=ivub, types=''.join(ivtype))
		self.__ivars = ivnames

		self.__indicator_map = {}
		for var, ivar in zip(dvnames, ivnames):
			self.__indicator_map[var] = ivar
			auxvars = [(ivar + name, 0, 1, btype) for name in ['a', 'b', 'c', 'd']]
			auxname, auxlb, auxub, auxtype = list(zip(*auxvars))
			a, b, c, d = auxname
			self.model.variables.add(names=auxname, lb=auxlb, ub=auxub, types=auxtype)

			# auxiliary constraints
			aux_lin = [
				([ivar, a], [1, 1]),
				([a, b], [-1, 1]),
				([ivar, c], [-1, 1]),
				([c, d], [-1, 1])
			]
			aux_names = ['C' + ivar + '_helper' + str(i) for i in range(4)]
			self.model.linear_constraints.add(lin_expr=aux_lin, senses='EGEG', rhs=[1, 0, 0, 0], names=aux_names)

			ind_lin = [([var], [1]), ([var, 'C'], [1, -1])]
			ind_names = ['C' + ivar + '_ind' + '1', 'C' + ivar + '_ind' + '2']
			self.model.indicator_constraints.add(lin_expr=ind_lin[0], sense='E', rhs=0, indvar=b, complemented=0,
												 name=ind_names[0])
			self.model.indicator_constraints.add(lin_expr=ind_lin[1], sense='G', rhs=0, indvar=d, complemented=0,
												 name=ind_names[1])

	def __add_efp_auxiliary_indicators(self):
		'''

		Returns:

		'''
		pass


	def __add_exclusivity_constraints(self):
		'''
		Adds constraints so that fluxes with two assigned dvars will only have one of the indicators active (flux must
		not be carried through both senses at once to avoid cancellation)
		-------

		'''
		lin_exprs = [([self.__indicator_map[var] for var in vlist], [1] * len(vlist)) for vlist in self.__dvars if
					 isinstance(vlist, tuple)]
		nc = len(lin_exprs)
		self.model.linear_constraints.add(lin_exprs, senses='L' * nc, rhs=[1] * nc,
										  names=['E' + str(i) for i in range(nc)])

	def __set_objective(self):
		'''
		Defines the objective for the optimization problem (Minimize the sum of all indicator variables)
		-------

		'''
		self.model.objective.set_sense(self.model.objective.sense.minimize)
		self.model.objective.set_linear(self.__objective_expression)

	# def __get_ivar_sum_vector(self, value_map):
	# 	return dict([[(svar.name for svar in var),
	# 				  sum(value_map[svar.name] for svar in var) if isinstance(var, list) else var.name,
	# 				  value_map[var.name]] for var in self.__ivars])

	def __integer_cut_count(self):
		'''

		Returns the amount of integer cuts added to the model
		-------

		'''

		return len(self.__integer_cuts)

	def __add_integer_cut(self, value_map, force_sol=False):
		'''
		Adds an integer cut based on a map of flux values (from a solution).

		Parameters
		----------
		value_map: A dictionary mapping solver variables with values
		force_sol: Boolean value indicating whether the solution is to be excluded or forced.
		-------

		'''
		lin_expr_vars = []
		counter = 0
		for varlist in self.__dvars:
			if isinstance(varlist, tuple):
				if sum(abs(value_map[self.__indicator_map[var]]) for var in varlist) > 0:
					lin_expr_vars.extend([self.__indicator_map[var] for var in varlist])
					counter += 1
			else:
				if abs(value_map[self.__indicator_map[varlist]]) > 0:
					lin_expr_vars.append(self.__indicator_map[varlist])
					counter += 1

		self.model.linear_constraints.add(names=['cut' + str(len(self.__integer_cuts))],
										  lin_expr=[[lin_expr_vars, [1] * len(lin_expr_vars)]], senses=['L'] if not force_sol else ['E'],
										  rhs=[counter - 1] if not force_sol else [counter])

	def set_size_constraint(self, start_at, equal=False):
		'''
		Defines the size constraint for the K-shortest algorithm.

		Parameters
		----------
		start_at: Size from which the solutions will be obtained.
		equal: Boolean indicating whether the solutions will match the size or can be higher than it.
		-------

		'''
		# TODO: Find a way to add a single constraint with two bounds.
		if self.SIZE_CONSTRAINT_NAME in self.model.linear_constraints.get_names():
			self.model.linear_constraints.set_rhs(self.SIZE_CONSTRAINT_NAME, start_at)
			self.model.linear_constraints.set_senses(self.SIZE_CONSTRAINT_NAME, 'E' if equal else 'G')
		else:
			lin_expr = [list(zip(*self.__objective_expression))]
			names = [self.SIZE_CONSTRAINT_NAME]
			senses = ['E' if equal else 'G']
			self.model.linear_constraints.add(lin_expr=lin_expr, names=names, senses=senses, rhs=[start_at])

	def get_model(self):
		'''

		Returns the solver instance.
		-------

		'''
		return self.model

	def __optimize(self):
		'''

		Optimizes the model and returns a single KShortestSolution instance for the model, adding an exclusion integer
		cut for it.
		-------

		'''
		try:
			self.model.solve()
			status = self.model.solution.get_status()
			value_map = dict(zip(self.model.variables.get_names(), self.model.solution.get_values()))
			if status > -1:
				sol = KShortestSolution(value_map, status, self.__indicator_map, self.__dvar_mapping)
				return sol
		except Exception as e:
			print(e)

	def __populate(self):
		'''
		Finds all feasible MIP solutions for the problem and returns them as a list of KShortestSolution instances.
		Returns
		-------

		'''
		#self.model.write('indicator_efmmodel.lp') ## For debug purposes
		sols = []
		self.model.populate_solution_pool()
		n_sols = self.model.solution.pool.get_num()
		for i in range(n_sols):
			value_map = dict(self.__solzip(self.model.solution.pool.get_values(i)))
			sol = KShortestSolution(value_map, None, self.__indicator_map, self.__dvar_mapping)
			sols.append(sol)
		for sol in sols:
			self.__add_integer_cut(sol.var_values())
		return sols

	def solution_iterator(self, maximum_amount=2**31-1):
		'''
		Generates a solution iterator. Each next call will yield a single solution. This method should be used to allow
		flexibility when enumerating EFMs for large problems. Since it uses the optimize routine, this may be slower in
		the longer term.
		-------

		'''
		i = 0
		self.reset_enumerator_state()
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
		'''
		Generates a solution iterator that yields a list of solutions. Each next call returns all EFMs for a single size
		starting from 1 up to max_size.
		Parameters
		----------
		max_size: The maximum solution size.

		Returns a list of KShortestSolution instances.
		-------

		'''
		self.reset_enumerator_state()
		for i in range(1, max_size + 1):
			print('Starting size',str(i))
			try:
				self.set_size_constraint(i, True)
				sols = self.populate_current_size()
				yield sols if sols is not None else []
			except Exception as e:
				print('No solutions or error occurred at size ', i)
				raise e

	def populate_current_size(self):
		'''
		Returns the solutions for the current size. Use the population_iterator method instead.
		-------

		'''
		sols = self.__populate()
		return sols

	def get_single_solution(self):
		'''

		Returns a single solution. Use the solution_iterator method instead.
		-------

		'''
		sol = self.__optimize()
		if sol is None:
			raise Exception('Solution is empty')
		self.__add_integer_cut(sol.var_values())
		return sol

	def reset_enumerator_state(self):
		'''
		Resets all integer cuts and size constraints.
		-------

		'''
		self.model.linear_constraints.delete(self.__integer_cuts)
		self.set_size_constraint(1)


class KShortestSolution(Solution):
	'''
	A Solution subclass that also contains attributes suitable for elementary flux modes such as non-cancellation sums
	of split reactions and reaction activity.
	'''
	SIGNED_INDICATOR_SUM = 'signed_indicator_map'
	SIGNED_VALUE_MAP = 'signed_value_map'

	def __init__(self, value_map, status, indicator_map, dvar_mapping, **kwargs):
		'''

		Parameters
		----------
		value_map: A dictionary mapping variable names with values
		status: See <Solution>
		indicator_map: A dictionary mapping indicators with
		dvar_mapping: A mapping between reaction indices and solver variables (Tuple[str] or str)
		kwargs: See <Solution>
		'''
		signed_value_map = {i: value_map[varlist[0]] - value_map[varlist[1]] if isinstance(varlist, tuple) else value_map[varlist] for
			i, varlist in dvar_mapping.items()}
		signed_indicator_map = {i: value_map[indicator_map[varlist[0]]] - value_map[indicator_map[varlist[1]]] if isinstance(varlist, tuple) else value_map[indicator_map[varlist]] for
			i, varlist in dvar_mapping.items()}
		super().__init__(value_map, status, **kwargs)
		self.set_attribute(self.SIGNED_VALUE_MAP, signed_value_map)
		self.set_attribute(self.SIGNED_INDICATOR_SUM, signed_indicator_map)

	def get_active_indicator_varids(self):
		'''

		Returns the indices of active indicator variables (maps with variables on the original stoichiometric matrix)
		-------

		'''
		return [k for k, v in self.attribute_value(self.SIGNED_INDICATOR_SUM).items() if v != 0]


class KShortestEFMAlgorithm(object):
	'''
	A higher level class to use the K-Shortest algorithm. This encompasses the standard routine for enumeration of EFMs.
	Requires a configuration defining an algorithm type. See <KShortestProperties>
	'''
	def  __init__(self, configuration, verbose=True):
		'''

		Parameters
		----------
		configuration: A KShortestProperties instance
		verbose: Boolean indicating whether to print useful messages while enumerating.
		'''
		assert configuration.__class__ == kp.KShortestProperties, 'Configuration class is not KShortestProperties'
		self.configuration = configuration
		self.verbose = verbose

	def __prepare(self, linear_system, excluded_sets, forced_sets):
		## TODO: Change this method's name
		'''
		Enumerates the elementary modes for a linear system
		Parameters
		----------
		linear_system: A KShortestCompatibleLinearSystem instance
		excluded_sets: Iterable[Tuple[Solution/Tuple]] with solutions to exclude from the enumeration
		forced_sets: Iterable[Tuple[Solution/Tuple]] with solutions to force

		Returns a list with solutions encoding elementary flux modes.
		-------

		'''
		assert self.configuration.has_required_properties(), "Algorithm configuration is missing required parameters."
		self.ksh = KShortestEnumerator(linear_system)
		if excluded_sets is not None:
			self.ksh.exclude_solutions(excluded_sets)
		if forced_sets is not None:
			self.ksh.force_solutions(forced_sets)

	def enumerate(self, linear_system, excluded_sets=None, forced_sets=None):
		'''
		Enumerates the elementary modes for a linear system
		Parameters
		----------
		linear_system: A KShortestCompatibleLinearSystem instance
		excluded_sets: Iterable[Tuple[Solution/Tuple]] with solutions to exclude from the enumeration
		forced_sets: Iterable[Tuple[Solution/Tuple]] with solutions to force

		Returns a list with solutions encoding elementary flux modes.
		-------

		'''
		enumerator = self.get_enumerator(linear_system, excluded_sets, forced_sets)
		return list(enumerator)


	def get_enumerator(self, linear_system, excluded_sets, forced_sets):
		"""
		Returns an iterator that yields one or multiple EFMs at each iteration, depending on the properties.
		Args:
		linear_system: A KShortestCompatibleLinearSystem instance
		excluded_sets: Iterable[Tuple[Solution/Tuple]] with solutions to exclude from the enumeration
		forced_sets: Iterable[Tuple[Solution/Tuple]] with solutions to force

		Returns a generator.

		"""
		self.__prepare(linear_system, excluded_sets, forced_sets)

		if self.configuration[kp.K_SHORTEST_MPROPERTY_METHOD] == kp.K_SHORTEST_METHOD_ITERATE:
			limit = self.configuration[kp.K_SHORTEST_OPROPERTY_MAXSOLUTIONS]
			if limit is None:
				limit = 1
				warnings.warn(Warning('You have not defined a maximum solution size for the enumeration process. Defaulting to 1.'))
			return ksh.solution_iterator(limit)

		elif self.configuration[kp.K_SHORTEST_MPROPERTY_METHOD] == kp.K_SHORTEST_METHOD_POPULATE:
			limit = self.configuration[kp.K_SHORTEST_OPROPERTY_MAXSIZE]
			if limit is None:
				warnings.warn(Warning('You have not defined a maximum size for the enumeration process. Defaulting to size 1.'))
				limit = 1
			return ksh.population_iterator(limit)
		else:
			raise Exception('Algorithm type is invalid! If you see this message, something went wrong!')
