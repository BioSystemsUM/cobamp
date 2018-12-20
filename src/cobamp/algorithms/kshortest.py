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

import cplex
from itertools import chain

from numpy.core.multiarray import concatenate, array, zeros

from cobamp.core.optimization import Solution, copy_cplex_model
from cobamp.core.linear_systems import IrreversibleLinearPatternSystem
from ..utilities.property_management import PropertyDictionary
import warnings

CPLEX_INFINITY = cplex.infinity
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

__kshortest_mandatory_properties = {
	K_SHORTEST_MPROPERTY_METHOD: [K_SHORTEST_METHOD_ITERATE, K_SHORTEST_METHOD_POPULATE]}

__kshortest_optional_properties = {
	K_SHORTEST_OPROPERTY_MAXSIZE: lambda x: x > 0 and isinstance(x, int),
	K_SHORTEST_OPROPERTY_MAXSOLUTIONS: lambda x: x > 0 and isinstance(x, int)
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
		super().__init__(__kshortest_mandatory_properties, __kshortest_optional_properties)

class KShortestEnumerator(object):
	"""
	Class implementing the k-shortest elementary flux mode algorithms. This is a lower level class implemented using the
	Cplex solver as base. Maybe in the future, this will be readapted for the optlang wrapper, if performance issues
	do not arise.
	"""

	ENUMERATION_METHOD_ITERATE = 'iterate'
	ENUMERATION_METHOD_POPULATE = 'populate'
	SIZE_CONSTRAINT_NAME = 'KShortestSizeConstraint'

	def __init__(self, linear_system):

		"""

		Parameters

		----------

			linear_system: A KShortestCompatibleLinearSystem/<LinearSystem> subclass

		"""

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
		self.is_efp_problem = isinstance(linear_system, IrreversibleLinearPatternSystem)
		# Setup k-shortest constraints
		self.__add_kshortest_indicators()
		if not self.is_efp_problem:
			self.__add_exclusivity_constraints()
		self.__size_constraint = None
		# TODO: change this to cplex notation
		self.__efp_auxiliary_map = None

		if self.is_efp_problem:
			self.__add_efp_auxiliary_constraints()

		self.__objective_expression = list(
			zip(list(self.__indicator_map.values()), [1] * len(self.__indicator_map.keys())))
		self.__set_objective()
		self.__integer_cuts = []
		self.__exclusion_cuts = []
		self.set_size_constraint(1)
		self.__current_size = 1
		self.model.write('ksmodel.lp')

	def __set_model_parameters(self):

		"""
		Internal method to set model parameters. This is based on the original MATLAB code by Von Kamp et al.

		-------
		"""

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

	##TODO: Make this more flexible in the future. 4GB of RAM should be enough but some problems might require more.

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
		for sol in sols:
			if isinstance(sol, KShortestSolution):
				self.__add_integer_cut(sol.var_values(), efp_cut=self.is_efp_problem)
			elif isinstance(sol, list) or isinstance(sol, tuple):
				ivars = [self.__indicator_map[k] for k in list(chain(*[self.__dvars[i] for i in sol]))]
				lin_expr = (ivars, [1] * len(ivars))
				sense = ['L']
				rhs = [len(sol) - 1]
				names = ['exclusion_cuts' + str(len(self.__exclusion_cuts))]
				self.model.linear_constraints.add(lin_expr=[lin_expr], senses=sense, rhs=rhs, names=names)

	def force_solutions(self, sols):
		"""
		Forces a set of reactions encoded as solutions to appear in the subsequent elementary modes to be calculated.

		Parameters

		----------

			sols: An Iterable containing list/tuples with active reaction combinations to exclude or Solution instances.

		-------

		"""
		for sol in sols:
			if isinstance(sol, KShortestSolution):
				self.__add_integer_cut(sol.var_values(), force_sol=True, efp_cut=self.is_efp_problem)
			elif isinstance(sol, list) or isinstance(sol, tuple):
				ivars = [self.__indicator_map[k] for k in list(chain(*[self.__dvars[i] for i in sol]))]
				lin_expr = (ivars, [1] * len(ivars))
				sense = ['E']
				rhs = [len(sol)]
				names = ['forced_cuts' + str(len(self.__exclusion_cuts))]
				self.model.linear_constraints.add(lin_expr=[lin_expr], senses=sense, rhs=rhs, names=names)

	def __add_kshortest_indicators(self):
		"""
		Adds indicator variable to a copy of the supplied linear problem.
		This uses the __dvars map to obtain a list of all variables and assigns an indicator to them.

		-------

		"""
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

	def __add_efp_auxiliary_constraints(self):
		self.__efp_auxiliary_map = {}
		btype = self.model.variables.type.binary
		for ind in self.__ivars:
			self.__efp_auxiliary_map[ind] = ''.join(['hv',ind])
		varnames = list(self.__efp_auxiliary_map.values())

		self.model.variables.add(names = varnames, types=btype*len(varnames))

		## Adding MILP2
		milp2_lex_template = lambda bv, hv: cplex.SparsePair(ind=[bv, hv],val=[1, -1])
		milp2_rhs, milp2_senses, milp2_names = [0]*len(varnames), 'G'*len(varnames), ['MILP2'+var for var in varnames]
		milp2_lin_expr = [milp2_lex_template(bv, hv) for bv, hv in self.__efp_auxiliary_map.items()]
		self.model.linear_constraints.add(lin_expr=milp2_lin_expr, rhs=milp2_rhs, senses=milp2_senses, names= milp2_names)

		## Adding MILP4
		milp4_lex = [
			(varnames, [1]*len(varnames))
		]
		self.model.linear_constraints.add(lin_expr=milp4_lex, senses='G', rhs=[1], names=['MILP4'])

	def __add_exclusivity_constraints(self):
		"""
		Adds constraints so that fluxes with two assigned dvars will only have one of the indicators active (flux must
		not be carried through both senses at once to avoid cancellation)

		-------

		"""
		lin_exprs = [([self.__indicator_map[var] for var in vlist], [1] * len(vlist)) for vlist in self.__dvars if
					 isinstance(vlist, tuple)]
		nc = len(lin_exprs)
		self.model.linear_constraints.add(lin_exprs, senses='L' * nc, rhs=[1] * nc,
										  names=['E' + str(i) for i in range(nc)])

	def __set_objective(self):
		"""
		Defines the objective for the optimization problem (Minimize the sum of all indicator variables)

		-------

		"""
		self.model.objective.set_sense(self.model.objective.sense.minimize)
		self.model.objective.set_linear(self.__objective_expression)

	# def __get_ivar_sum_vector(self, value_map):
	# 	return dict([[(svar.name for svar in var),
	# 				  sum(value_map[svar.name] for svar in var) if isinstance(var, list) else var.name,
	# 				  value_map[var.name]] for var in self.__ivars])

	def __integer_cut_count(self):
		"""

		Returns the amount of integer cuts added to the model

		-------

		"""

		return len(self.__integer_cuts)

	def __add_integer_cut(self, value_map, force_sol=False, efp_cut=False):
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

		lin_expr_vars = []
		counter = 0
		for varlist in self.__dvars:
			if isinstance(varlist, tuple):
				if sum(abs(value_map[self.__indicator_map[var]]) for var in varlist) > 0:
					if efp_cut:
						ivars = [self.__indicator_map[var] for var in varlist if value_map[var] > 0]
						lin_expr_vars.extend([self.__efp_auxiliary_map[v] for v in ivars])
					else:
						ivars = [self.__indicator_map[var] for var in varlist]
					lin_expr_vars.extend(ivars)
					counter += 1
			else:
				if abs(value_map[self.__indicator_map[varlist]]) > 0:
					lin_expr_vars.append(self.__indicator_map[varlist])
					if efp_cut:
						lin_expr_vars.append(self.__efp_auxiliary_map[self.__indicator_map[varlist]])
					counter += 1

		self.model.linear_constraints.add(names=['cut' + str(len(self.__integer_cuts))],
										  lin_expr=[(lin_expr_vars, [1]*len(lin_expr_vars))],
										  senses=['L'] if not force_sol else ['E'],
										  rhs=[counter - (1 * int(not efp_cut))] if not force_sol else [counter])

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
		if self.SIZE_CONSTRAINT_NAME in self.model.linear_constraints.get_names():
			self.model.linear_constraints.set_rhs(self.SIZE_CONSTRAINT_NAME, start_at)
			self.model.linear_constraints.set_senses(self.SIZE_CONSTRAINT_NAME, 'E' if equal else 'G')
		else:
			lin_expr = [list(zip(*self.__objective_expression))]
			names = [self.SIZE_CONSTRAINT_NAME]
			senses = ['E' if equal else 'G']
			self.model.linear_constraints.add(lin_expr=lin_expr, names=names, senses=senses, rhs=[start_at])

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
			self.model.solve()
			status = self.model.solution.get_status()
			value_map = dict(zip(self.model.variables.get_names(), self.model.solution.get_values()))
			if status > -1:
				sol = KShortestSolution(value_map, status, self.__indicator_map, self.__dvar_mapping)
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
		self.model.populate_solution_pool()
		n_sols = self.model.solution.pool.get_num()
		for i in range(n_sols):
			value_map = dict(self.__solzip(self.model.solution.pool.get_values(i)))
			sol = KShortestSolution(value_map, None, self.__indicator_map, self.__dvar_mapping)
			sols.append(sol)
		for sol in sols:
			self.__add_integer_cut(sol.var_values(), efp_cut=self.is_efp_problem)
		return sols

	def solution_iterator(self, maximum_amount=2 ** 31 - 1):
		"""
		Generates a solution iterator. Each next call will yield a single solution. This method should be used to allow
		flexibility when enumerating EFMs for large problems. Since it uses the optimize routine, this may be slower in
		the longer term.

		-------

		"""
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
				print('Enumeration ended:', e.with_traceback())
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
		self.reset_enumerator_state()
		for i in range(1, max_size + 1):
			print('Starting size', str(i))
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
		self.model.linear_constraints.delete(self.__integer_cuts)
		self.set_size_constraint(1)


class KShortestSolution(Solution):
	"""
	A Solution subclass that also contains attributes suitable for elementary flux modes such as non-cancellation sums
	of split reactions and reaction activity.
	"""
	SIGNED_INDICATOR_SUM = 'signed_indicator_map'
	SIGNED_VALUE_MAP = 'signed_value_map'

	def __init__(self, value_map, status, indicator_map, dvar_mapping, **kwargs):
		"""

		Parameters

		----------

			value_map: A dictionary mapping variable names with values

			status: See <Solution>

			indicator_map: A dictionary mapping indicators with

			dvar_mapping: A mapping between reaction indices and solver variables (Tuple[str] or str)

			kwargs: See <Solution>

		"""
		signed_value_map = {
			i: value_map[varlist[0]] - value_map[varlist[1]] if isinstance(varlist, tuple) else value_map[varlist] for
			i, varlist in dvar_mapping.items()}
		signed_indicator_map = {
			i: value_map[indicator_map[varlist[0]]] - value_map[indicator_map[varlist[1]]] if isinstance(varlist,
																										 tuple) else
			value_map[indicator_map[varlist]] for
			i, varlist in dvar_mapping.items()}
		super().__init__(value_map, status, **kwargs)
		self.set_attribute(self.SIGNED_VALUE_MAP, signed_value_map)
		self.set_attribute(self.SIGNED_INDICATOR_SUM, signed_indicator_map)

	def get_active_indicator_varids(self):
		"""

		Returns the indices of active indicator variables (maps with variables on the original stoichiometric matrix)

		-------

		"""
		return [k for k, v in self.attribute_value(self.SIGNED_INDICATOR_SUM).items() if v != 0]


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
		self.ksh = KShortestEnumerator(linear_system)
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
			limit = self.configuration[kp.K_SHORTEST_OPROPERTY_MAXSOLUTIONS]
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