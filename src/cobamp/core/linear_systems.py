import abc
import warnings
from collections import OrderedDict
from itertools import chain
from multiprocessing import cpu_count

import numpy as np
import optlang
from optlang.symbolics import Zero

CUSTOM_DEFAULT_SOLVER = None
SOLVER_ORDER = ['CPLEX', 'GUROBI', 'GLPK', 'MOSEK', 'SCIPY']


def get_default_solver():
	if CUSTOM_DEFAULT_SOLVER:
		return CUSTOM_DEFAULT_SOLVER
	else:
		for solver in SOLVER_ORDER:
			if optlang.list_available_solvers()[solver]:
				return solver


def get_solver_interfaces():
	s = {}
	for solver, status in optlang.list_available_solvers().items():
		if status:
			s[solver] = eval('optlang.' + solver.lower() + '_interface')
	return s


SOLVER_INTERFACES = get_solver_interfaces()
DEFAULT_SOLVER = get_default_solver()

VAR_CONTINUOUS, VAR_INTEGER, VAR_BINARY = ('continuous', 'integer', 'binary')
VAR_TYPES = (VAR_CONTINUOUS, VAR_INTEGER, VAR_BINARY)
VARIABLE, CONSTRAINT = 'var', 'const'
SENSE_MINIMIZE, SENSE_MAXIMIZE = ('min', 'max')


def fwd_irrev(lb, ub):
	"""
	Args:
	 lb:
	 ub:
	"""
	return ((lb >= 0) and (ub >= 0)).astype(int)


def bak_irrev(lb, ub):
	"""
	Args:
	 lb:
	 ub:
	"""
	return ((lb < 0) and (ub <= 0)).astype(int)


def fix_backwards_irreversible_reactions(S, lb, ub):
	"""
	Args:
	 S:
	 lb:
	 ub:
	"""
	S = np.array(S)
	lb, ub = np.array(lb), np.array(ub)
	fwd_irrev_index, bak_irrev_index = [[i for i in range(S.shape[1]) if fx(lb[i], ub[i])] for fx in
										[fwd_irrev, bak_irrev]]
	# irrev = np.union1d(fwd_irrev_index, bak_irrev_index)

	if len(bak_irrev_index) > 0:
		S[:, bak_irrev_index] = -S[:, bak_irrev_index]
		ub_temp = ub[bak_irrev_index]
		ub[bak_irrev_index] = -lb[bak_irrev_index]
		lb[bak_irrev_index] = -ub_temp

	return S, lb, ub, fwd_irrev_index, bak_irrev_index


def make_irreversible_model(S, lb, ub):
	# lb, ub = np.array(lb), np.array(ub)
	# # fwd_irrev = lambda lb, ub: (lb >= 0) and (ub >= 0)
	# # bak_irrev = lambda lb, ub: (lb < 0) and (ub <= 0)
	#
	# fwd_irrev_index, bak_irrev_index = [[i for i in range(S.shape[1]) if fx(lb[i], ub[i])] for fx in [fwd_irrev, bak_irrev]]
	"""
	Args:
	 S:
	 lb:
	 ub:
	"""
	S, lb, ub, fwd_irrev_index, bak_irrev_index = fix_backwards_irreversible_reactions(S, lb, ub)

	irrev = np.union1d(fwd_irrev_index, bak_irrev_index)

	# irrev = np.array([i for i in range(S.shape[1]) if not (lb[i] < 0 and ub[i] > 0)])
	rev = np.array([i for i in range(S.shape[1]) if i not in irrev])

	# if len(bak_irrev_index) > 0:
	# 	S[:, bak_irrev_index] = -S[:, bak_irrev_index]
	# 	ub_temp = ub[bak_irrev_index]
	# 	ub[bak_irrev_index] = -lb[bak_irrev_index]
	# 	lb[bak_irrev_index] = -ub_temp

	if len(rev) > 0:
		Sr = S[:, rev]
		S_new = np.hstack([S, -Sr])
	else:
		S_new = S.copy()
	offset = S.shape[1]
	rx_mapping = {k: v if k in irrev else [v] for k, v in dict(zip(range(offset), range(offset))).items()}
	for i, rx in enumerate(rev):
		rx_mapping[rx].append(offset + i)
	rx_mapping = OrderedDict([(k, tuple(v)) if isinstance(v, list) else (k, v) for k, v in rx_mapping.items()])

	nlb, nub = np.zeros(S_new.shape[1]), np.zeros(S_new.shape[1])
	for orig_rx, new_rx in rx_mapping.items():
		if isinstance(new_rx, tuple):
			nub[new_rx[0]] = abs(lb[orig_rx])
			nub[new_rx[1]] = ub[orig_rx]
		else:
			nlb[new_rx], nub[new_rx] = lb[orig_rx], ub[orig_rx]

	return S_new, nlb, nub, rx_mapping


class LinearSystem():
	"""An abstract class defining the template for subclasses implementing linear
	systems that can be used with optimizers such as CPLEX and passed onto other
	algorithms supplied with the package.

	Must instantiate the following variables:

	 S: System of linear equations represented as a n-by-m ndarray, preferrably
	 with dtype as float or int

	 __model: Linear model as an instance of the solver.
	"""
	__metaclass__ = abc.ABCMeta
	model = None

	def was_built(self):
		if self.model != None:
			return (len(self.model.variables) != 0) or (len(self.model.constraints) != 0)
		else:
			return False

	def select_solver(self, solver=None):
		"""
		Args:
		solver:
		"""
		if not solver:
			solver = get_default_solver()
			self.interface = SOLVER_INTERFACES[solver]
		else:
			if solver in SOLVER_INTERFACES:
				self.interface = SOLVER_INTERFACES[solver]

			else:
				raise Exception(solver, 'Solver not found. Choose from ', str(list(SOLVER_INTERFACES.keys())))
		self.solver = solver

	def get_configuration(self):
		return self.model.configuration

	def set_default_configuration(self):
		cfg = self.model.configuration

		try:
			cfg.tolerances.feasibility = 1e-9
			cfg.tolerances.optimality = 1e-9
			cfg.tolerances.integrality = 1e-9
			cfg.lpmethod = 'auto'
			cfg.presolve = True

		except:
			print('Could not set parameters with this solver')

	def get_constraint_bounds(self, constraints=None):
		"""
		Args:
		  constraints:
		"""
		return [(c.lb, c.ub) for c in (self.model.constraints if constraints == None else constraints)]

	def get_constraint_matrices(self, constraints=None, vars=None):
		"""
		Args:
		  constraints:
		  vars:
		"""
		return [self.get_system_matrix(constraints, vars)] + list(zip(*self.get_constraint_bounds(constraints)))

	def get_system_matrix(self, constraints=None, vars=None):
		"""
		Args:
		  constraints:
		  vars:
		"""
		var_list = self.model.variables if vars == None else vars
		cons_list = self.model.constraints if constraints == None else constraints
		var_dict = dict(list(zip(*list(zip(*enumerate(var_list)))[::-1])))
		cnst = cons_list

		def row_factory(mask, values):
			row = np.zeros([1, len(var_list)])
			# print(len(mask), len(values), row.shape)
			row[:, mask] = values
			return row

		def get_mask_value_list(constraint):
			if constraint.indicator_variable == None:
				return list(zip(
					*[(var_dict[var], value) for var, value in constraint.get_linear_coefficients(var_list).items()]))
			else:
				return [(0, 1), (0, 0)]

		A = np.vstack([row_factory(*get_mask_value_list(c)) for c in cnst])
		return A

	def set_number_of_threads(self, n_threads=0):
		"""Defines the amount of threads available for the solver to use. :param
		n_threads: Number of threads available to the solver. Set to 0 if default
		parameters are needed :return:

		Args:
		  n_threads:
		"""
		cpu_c = cpu_count()
		is_positive = n_threads >= 0
		actual_threads = n_threads

		if not is_positive:
			warnings.warn('Threads cannot be set to negative values. Using default values')
			actual_threads = 0

		if n_threads > cpu_c:
			warnings.warn('User-defined number of threads exceeds the amount of physical threads')

		if self.solver == 'CPLEX':
			self.model.problem.parameters.threads.set(actual_threads)
		elif self.solver == 'GUROBI':
			self.model.problem.Params.Threads = actual_threads
		else:
			warnings.warn('Could not set threads for ' + str(self.solver) + ' instance. Not yet implemented!')

	def set_working_memory_limit(self, workmem):
		"""Defines the amount of memory available for the solver to use. Use this at
		your own peril! :param n_threads: Memory in MegaBytes available to the solver
		:return:

		Args:
		  workmem:
		"""
		is_positive = workmem >= 0
		actual_mem = workmem

		if not is_positive:
			warnings.warn('Threads cannot be set to negative values. Using default values')

		else:
			if self.solver == 'CPLEX':
				self.model.problem.parameters.workmem.set(actual_mem)
			elif self.solver == 'GUROBI':
				self.model.problem.Params.NodefileStart = actual_mem / 1024
			else:
				warnings.warn('Could not set threads for ' + str(self.solver) + ' instance. Not yet implemented!')

	@abc.abstractmethod
	def build_problem(self):
		"""Builds a CPLEX model with the constraints specified in the constructor
		arguments. This method must be implemented by any <LinearSystem>. Refer to the
		:ref:`constructor <self.__init__>` -------
		"""
		pass

	def get_model(self):
		"""Returns the model instance. Must call <self.build_problem()> to return a
		CPLEX model.

		-------
		"""
		return self.model

	def get_stoich_matrix_shape(self):
		"""Returns a tuple with the shape (rows, columns) of the supplied
		stoichiometric matrix.

		-------
		"""
		return self.S.shape

	# def empty_constraint(self, b_lb, b_ub, dummy_var=Variable(name='dummy', lb=0, ub=0)):
	def empty_constraint(self, b_lb, b_ub, **kwargs):
		"""
		Args:
		  b_lb:
		  b_ub:
		  **kwargs:
		"""
		return self.interface.Constraint(Zero, lb=b_lb, ub=b_ub, **kwargs)

	def dummy_variable(self):
		return self.interface.Variable(name='dummy', lb=0, ub=0)

	def populate_model_from_matrix(self, S, var_types, lb, ub, b_lb, b_ub, var_names, only_nonzero=False,
								   indicator_rows=None):
		"""
		Args:
		  S: Two-dimensional np.ndarray instance
		  var_types: list or tuple with length equal to the amount of columns of S
			  containing variable types (e.g.
		  lb: list or tuple with length equal to the amount of columns of S
			  containing the lower bounds for all variables
		  ub: list or tuple with length equal to the amount of columns of S
			  containing the upper bounds for all variables
		  b_lb: list or tuple with length equal to the amount of rows of S
			  containing the lower bound for the right hand
		  b_ub: list or tuple with length equal to the amount of rows of S
			  containing the upper bound for the right hand
		  var_names: string identifiers for the variables
		  only_nonzero:
		  indicator_rows:
		"""
		self.add_variables_to_model(var_names, lb, ub, var_types)
		self.add_rows_to_model(S, b_lb, b_ub, only_nonzero, indicator_rows)

	def populate_constraints_from_matrix(self, S, constraints, vars, only_nonzero=False):
		"""
		Args:
		  S: Two-dimensional np.ndarray instance
		  constraints (side of all):
		  vars: list of variable instances
		  only_nonzero:
		"""
		if not only_nonzero:
			coef_list = [{vars[j]: S[i, j] for j in range(S.shape[1])} for i in range(S.shape[0])]
		else:
			coef_list = [{vars[j]: S[i, j] for j in np.nonzero(S[i, :])[0]} for i in range(S.shape[0])]

		for coefs, constraint in zip(coef_list, constraints):
			if constraint.indicator_variable is None:
				constraint.set_linear_coefficients(coefs)

		for coefs, constraint in zip(coef_list, constraints):
			if constraint.indicator_variable is None:
				if len(coefs) > 0:
					constraint.set_linear_coefficients(coefs)

		self.model.update()

	def add_rows_to_model(self, S_new, b_lb, b_ub, only_nonzero=False, indicator_rows=None, vars=None, names=None):
		"""
		Args:
		  S_new:
		  b_lb:
		  b_ub:
		  only_nonzero:
		  indicator_rows:
		  vars:
		  names:
		"""
		if not vars:
			vars = self.model.variables
		# dummy = self.dummy_variable()
		if names != None:
			constraints = [self.empty_constraint(b_lb[i], b_ub[i], name=names[i]) for i in range(S_new.shape[0])]
		else:
			constraints = [self.empty_constraint(b_lb[i], b_ub[i]) for i in range(S_new.shape[0])]
		if indicator_rows:
			for row, var_idx, complement in indicator_rows:
				constraints[row] = self.interface.Constraint(
					sum(S_new[row, i] * vars[i] for i in S_new[row, :].nonzero()[0]), lb=b_lb[row], ub=b_ub[row],
					indicator_variable=vars[var_idx], active_when=complement)
		self.model.add(constraints, sloppy=True)

		self.model.update()

		self.populate_constraints_from_matrix(S_new, constraints, vars, only_nonzero)
		# self.model.update()
		# self.model.remove(dummy)
		self.model.update()
		return constraints

	def remove_from_model(self, index, what):
		"""
		Args:
		  index:
		  what:
		"""
		container = self.model.variables if what == VARIABLE else self.model.constraints if what == CONSTRAINT else None
		if type(index) not in (list, tuple):
			index = [index]
		for i in index:
			self.model.remove(container[i])

		if what == VARIABLE:
			self.S = np.delete(self.S, index, 1)
		elif what == CONSTRAINT:
			self.S = np.delete(self.S, index, 0)

		self.model.update()

	def add_columns_to_model(self, S_new, var_names, lb, ub, var_types):
		"""
		Args:
		  S_new:
		  var_names:
		  lb:
		  ub:
		  var_types:
		"""
		vars = self.add_variables_to_model(var_names, lb, ub, var_types)

		self.populate_constraints_from_matrix(S_new, self.model.constraints, vars)

	def add_variables_to_model(self, var_names, lb, ub, var_types):

		"""
		Args:
		  var_names:
		  lb:
		  ub:
		  var_types:
		"""
		if isinstance(var_types, str):
			var_types = [var_types] * len(lb)

		vars = [self.interface.Variable(name=name, lb=lbv, ub=ubv, type=t) for name, lbv, ubv, t in
				zip(var_names, lb, ub, var_types)]
		self.model.add(vars)
		self.model.update()

		return vars

	def get_stuff(self, what, index):
		"""
		Args:
		  what:
		  index:
		"""
		container = self.model.variables if what == VARIABLE else self.model.constraints if what == CONSTRAINT else None
		if container == None:
			raise ValueError('`what` parameter requires a string (either `var` or `const`)')

		return [container[k] for k in index]

	def set_objective(self, coefficients, minimize, vars=None):
		"""
		Args:
		  coefficients:
		  minimize:
		  vars:
		"""
		if not vars:
			vars = self.model.variables
		lcoefs = {k: v for k, v in zip(vars, coefficients) if v != 0}
		if len(lcoefs) == 0:
			lcoefs = {k: 0 for k in vars}
		try:
			dummy = self.dummy_variable()
			new_obj = self.interface.Objective(dummy)
			self.model.objective = new_obj
			new_obj.set_linear_coefficients(lcoefs)
		except Exception as e:
			raise e
		finally:
			self.model.remove(dummy)
			self.model.objective.direction = SENSE_MINIMIZE if minimize else SENSE_MAXIMIZE
			self.model.update()

	def set_variable_bounds(self, vars, lb, ub):
		"""
		Args:
		  vars:
		  lb:
		  ub:
		"""
		for var, ulb, uub in zip(vars, lb, ub):
			var.set_bounds(ulb, uub)
		self.model.update()

	def set_constraint_bounds(self, constraints, lb, ub):
		"""
		Args:
		  constraints:
		  lb:
		  ub:
		"""
		for c, ulb, uub in zip(constraints, lb, ub):
			b = ulb, uub
			lb_is_greater = b[0] > c.ub if ((c.ub is not None) and (b[0] is not None)) else False
			ub_is_lower = b[1] < c.lb if ((c.lb is not None) and (b[1] is not None)) else False
			if lb_is_greater:
				c.ub = b[1]
				c.lb = b[0]
			elif ub_is_lower:
				c.lb = b[0]
				c.ub = b[1]
			else:
				c.lb, c.ub = b
		self.model.update()

	def set_variable_types(self, vars, types):
		"""
		Args:
		  vars:
		  types:
		"""
		if isinstance(types, str):
			for var in vars:
				var.type = types
		else:
			for var, typ in zip(vars, types):
				if typ in VAR_TYPES:
					var.type = typ
				else:
					warnings.warn('Invalid variable type: ' + typ)
		self.model.update()

	def write_to_lp(self, filename):
		"""
		Args:
		  filename:
		"""
		with open(filename, 'w') as f:
			f.write(self.model.to_lp())


class KShortestCompatibleLinearSystem(LinearSystem):
	"""Abstract class representing a linear system that can be passed as an
	argument for the KShortestAlgorithm class. Subclasses must instantiate the
	following variables:

	 __dvar_mapping: A dictionary mapping reaction indexes with variable names

	 __dvars: A list of variable names (str) or Tuple[str] if two linear system
	 variables represent a single flux. Should be kept as type `list` to
	 maintain order.

	 __c: str representing the variable to be used as constant for indicator
	 constraints
	"""
	dvar_mapping = None
	dvars = None
	c = None

	def get_dvar_mapping(self):
		"""Returns a dictionary mapping flux indices with variable(s) on the optimization problem
		-------
		"""
		return self.dvar_mapping

	def get_dvars(self):
		"""Returns a list of variables (str or Tuple[str]) with similar order to that of the fluxes passed to the system.
		-------
		"""
		return self.dvars

	def add_c_variable(self):
		self.c = self.add_variables_to_model(['C'], [1], [None], VAR_CONTINUOUS)[0]

	def get_c_variable(self):
		return self.model.variables['C']


class GenericLinearSystem(LinearSystem):
	"""Class representing a generic system of linear (in)equations Used as
	arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, var_types, lb, ub, b_lb, b_ub, var_names, solver=None):
		"""Constructor for GenericLinearSystem

		Parameters

		  model: Optlang model S: Two-dimensional np.ndarray instance var_types:
		  list or tuple with length equal to the amount of columns of S containing
		  variable types (e.g. VAR_CONTINUOUS) lb: list or tuple with length equal
		  to the amount of columns of S containing the lower bounds for all
		  variables ub: list or tuple with length equal to the amount of columns of
		  S containing the upper bounds for all variables b_lb: list or tuple with
		  length equal to the amount of rows of S containing the lower bound for the
		  right hand side of all constraints b_ub: list or tuple with length equal
		  to the amount of rows of S containing the upper bound for the right hand
		  side of all constraints var_names: string identifiers for the variables

		Args:
		  S:
		  var_types:
		  lb:
		  ub:
		  b_lb:
		  b_ub:
		  var_names:
		  solver:
		"""
		self.select_solver(solver)
		self.model = self.interface.Model()
		self.set_default_configuration()
		self.S, self.lb, self.ub, self.b_lb, self.b_ub, self.var_types = S, lb, ub, b_lb, b_ub, var_types

		self.names = var_names if var_names is not None else ['v' + str(i) for i in range(S.shape[1])]

	def build_problem(self):
		self.populate_model_from_matrix(self.S, self.var_types, self.lb, self.ub, self.b_lb, self.b_ub, self.names,
										True)


class SteadyStateLinearSystem(GenericLinearSystem):
	"""Class representing a steady-state biological system of metabolites and
	reactions without dynamic parameters Used as arguments for various algorithms
	implemented in the package.
	"""

	def __init__(self, S, lb, ub, var_names, solver=None):
		"""Constructor for SimpleLinearSystem

		Parameters

		----------

		  S: Stoichiometric matrix represented as a n-by-m ndarray, preferrably with
		  dtype as float or int

		  lb: ndarray or list containing the lower bounds for all n fluxes

		  ub: ndarray or list containing the lower bounds for all n fluxes

		  var_names: - optional - ndarray or list containing the names for each flux

		Args:
		  S:
		  lb:
		  ub:
		  var_names:
		  solver:
		"""
		m, n = S.shape
		self.lb, self.ub = lb, ub
		super().__init__(S, VAR_CONTINUOUS, lb, ub, [0] * m, [0] * m, var_names, solver=solver)


def prepare_irreversible_system(self, S, lb, ub, non_consumed, consumed, produced, solver, force_bounds, add_c=True):
	"""
	Args:
	 self:
	 S:
	 lb:
	 ub:
	 non_consumed:
	 consumed:
	 produced:
	 solver:
	 force_bounds:
	"""
	self.select_solver(solver)

	# lb = [0 if i in irrev else -1 for i in range(S.shape[1])]
	# ub = [1] * S.shape[1]
	# if -1 in lb:
	S, lb, ub, fwd_irrev, bak_irrev = fix_backwards_irreversible_reactions(S, lb, ub)

	Si, lbi, ubi, rev_mapping = make_irreversible_model(S, lb, ub)
	fwd_names = ['V' + str(i) if not isinstance(v, list) else 'V' + str(i) + 'F' for i, v in rev_mapping.items()]
	bwd_names = ['V' + str(i) + 'R' for i, v in rev_mapping.items() if isinstance(v, (list, tuple))]

	# else:
	# 	Si, lbi, ubi = S, lb, ub
	# 	fwd_names = ['V' + str(i) for i in range(Si.shape[1])]
	# 	bwd_names = []
	# 	rev_mapping = {i:i for i in range(Si.shape[1])}

	lbi = [0] * len(lbi) + [1] if add_c else []
	ubi = [None] * len(ubi) + [None] if add_c else []

	for k, tup in force_bounds.items():
		if isinstance(rev_mapping[k], (list, tuple)):
			inds = rev_mapping[k]
			trev = (abs(ub[k]) if ub[k] < 0 else 0, abs(lb[k]) if lb[k] < 0 else 0)
			tfwd = (abs(lb[k]) if lb[k] > 0 else 0, abs(ub[k]) if ub[k] > 0 else 0)
			assert sum([pair[0] < pair[1] for pair in (trev, tfwd)]) == 2, 'force_bounds contains invalid values'
			for i, ntup in zip(inds, (tfwd, trev)):
				lbi[i], ubi[i] = ntup
		else:
			assert tup[0] < tup[1], 'force_bounds contains invalid values'
			lbi[k], ubi[k] = tup

	## TODO: remove this line - added for debugging
	# ubi = [10000 if ((k == None) or (k >= 10000)) else k for k in ubi]

	self.rev_mapping = rev_mapping
	self.__ivars = None
	self.__ss_override = [(nc, 'G', 0) for nc in non_consumed] + [(p, 'G', 1) for p in produced] + [(c, 'L', -1) for
																									c in consumed]
	Si = np.hstack([Si, np.zeros((Si.shape[0], 1))]) if add_c else Si
	b_lb, b_ub = [0] * Si.shape[0], [0] * Si.shape[0]
	## TODO: Maybe allow other values to be provided for constraint relaxation/tightening

	for id, _, v in self.__ss_override:
		b_lb[id], b_ub[id] = (v, None) if v >= 0 else (None, v)

	return Si, lbi, ubi, b_lb, b_ub, fwd_names, bwd_names, rev_mapping


class IrreversibleLinearSystem(KShortestCompatibleLinearSystem, GenericLinearSystem):
	"""Class representing a steady-state biological system of metabolites and
	reactions without dynamic parameters. All irreversible reactions are split into
	their forward and backward components so every lower bound is 0. Used as
	arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, lb, ub, non_consumed=(), consumed=(), produced=(), solver=None, force_bounds={}):
		"""
		Args:
		  S (Stoichiometric matrix represented as a n-by-m ndarray, preferrably with dtype as float or int):
		  lb:
		  ub:
		  non_consumed (An Iterable[int] or ndarray containing the indices of external metabolites not consumed in the):
		  consumed (An Iterable[int] or ndarray containing the indices of external metabolites guaranteed to be produced.):
		  produced (An Iterable[int] or ndarray containing the indices of external metabolites guaranteed to be consumed.):
		  solver (String specifying the LP solver to be used):
		  force_bounds (Dictionary mapping indexes with 2-tuples with lower and upper bounds):
		"""
		Si, lbi, ubi, b_lb, b_ub, fwd_names, bwd_names, rev_mapping = \
			prepare_irreversible_system(self, S, lb, ub, non_consumed, consumed, produced, solver, force_bounds)

		super().__init__(Si, VAR_CONTINUOUS, lbi, ubi, b_lb, b_ub, fwd_names + bwd_names + ['C'], solver=solver)

		self.dvars = list(range(S.shape[1] + len(bwd_names)))
		self.dvar_mapping = dict(rev_mapping)


class IrreversibleLinearPatternSystem(IrreversibleLinearSystem):
	## TODO: Code + docstrings. Do not use this yet!
	def __init__(self, S, lb, ub, subset, **kwargs):
		"""
		Args:
		  S:
		  lb:
		  ub:
		  subset:
		  **kwargs:
		"""
		super().__init__(S, lb, ub, **kwargs)
		self.subset = subset
		fwds, revs = [], []
		dvm_new = {}
		for r in self.subset:
			if not isinstance(self.rev_mapping[r], int):
				fid, rid = self.dvar_mapping[r]
				dvm_new[r] = (len(fwds), len(subset) + len(revs))
				fwds.append(fid)
				revs.append(rid)
			else:
				dvm_new[r] = len(fwds)
				fwds.append(self.dvar_mapping[r])

		self.dvar_mapping = dvm_new
		self.dvars = fwds + revs

	def build_problem(self):
		super().build_problem()

class GeneticDualLinearSystem(KShortestCompatibleLinearSystem, GenericLinearSystem):
	def __init__(self, S, lb, ub, G, T, b, solver=None):
		self.select_solver(solver)
		Si, lbi, ubi, b_lb, b_ub, fwd_names, bwd_names, rev_mapping = \
			prepare_irreversible_system(self, S, lb, ub, [], [], [], solver, {}, add_c=False)
		#
		# list_bi = []
		# list_ti = []
		# # fix T and b
		#

		self.__ivars = None
		self.__c = "C"
		Gi = np.zeros([G.shape[0], Si.shape[1]])
		for i, tup in rev_mapping.items():
			Gi[:,tup] = np.vstack([G[:,i], G[:,i]]).reshape(-1, 2) if isinstance(tup, (tuple,list)) else G[:,i]
		super().__init__(*self.generate_dual_problem(Si, Gi, T, b), solver=solver)

		self.dvars = list([Si.shape[0]+i for i in range(G.shape[0])])
		self.dvar_mapping = {i: i for i in range(len(self.dvars))}

	def generate_dual_problem(self, S, G, T, b):
		m, n = S.shape
		Sxi = S.T
		Ii = G.T
		Ti = T.T

		veclens = [("u", m), ("v", G.shape[0]), ("w", T.shape[0])]

		var_prop_list = [[(pref + str(i), 0 if pref != "u" else None, None) for i in range(n)] for
						pref, n in
						veclens]

		Sdi = np.hstack([Sxi, Ii, Ti, np.zeros([Sxi.shape[0], 1])])
		Sd = np.vstack([Sdi, np.zeros([1, Sdi.shape[1]])])

		#names, v_lb, v_ub = list(zip(*list(chain(u, v, w))))
		names, v_lb, v_ub = list(zip(*chain(*var_prop_list)))

		names = list(names) + ['C']
		v_lb = list(v_lb) + [1]
		v_ub = list(v_ub) + [None]

		w_idx = [m + Ii.shape[1] + i for i in range(len(b))]
		Sd[-1, w_idx] = b
		Sd[-1, -1] = 1

		b_ub = np.hstack([np.array([None] * Sdi.shape[0]), np.array([0])])
		b_lb = np.array([0] * (Sd.shape[0]), dtype=object)
		b_ub[-1] = 0
		b_lb[-1] = None

		return Sd, VAR_CONTINUOUS, v_lb, v_ub, b_lb, b_ub, names



class DualLinearSystem(KShortestCompatibleLinearSystem, GenericLinearSystem):
	"""Class representing a dual system based on a steady-state metabolic network
	whose elementary flux modes are minimal cut sets for use with the KShortest
	algorithms. Based on previous work by Ballerstein et al. and Von Kamp et al.

	[1] von Kamp, A., & Klamt, S. (2014). Enumeration of smallest intervention
	strategies in genome-scale metabolic networks. PLoS computational biology,
	10(1), e1003378. [2] Ballerstein, K., von Kamp, A., Klamt, S., & Haus, U. U.
	(2011). Minimal cut sets in a metabolic network are elementary modes in a dual
	network. Bioinformatics, 28(3), 381-387.
	"""

	def __init__(self, S, lb, ub, T, b, solver=None, alt_ident=None):
		"""Parameters

		----------

		  S: Stoichiometric matrix represented as a n-by-m ndarray, preferrably with
		  dtype as float or int irrev: An Iterable[int] or ndarray containing the
		  indices of irreversible reactions

		  T: Target matrix as an ndarray. Should have c-by-n dimensions (c -
		  #constraints; n - #fluxes)

		  b: Inhomogeneous bound values as a list or 1D ndarray of c size n.

		Args:
		  S:
		  lb:
		  ub:
		  T:
		  b:
		  solver:
		"""
		self.select_solver(solver)

		S, lb, ub, fwd_irrev, bak_irrev = fix_backwards_irreversible_reactions(S, lb, ub)

		irrev = np.union1d(fwd_irrev, bak_irrev).astype(int)

		self.__ivars = None
		self.S, self.irrev, self.T, self.b = S, irrev, T, b
		self.__c = "C"
		super().__init__(*self.generate_dual_problem(S, irrev, T, b, alt_ident), solver=solver)

		offset = S.shape[0]
		self.dvars = list(range(offset, offset + (S.shape[1] * 2)))

		self.dvar_mapping = {i: (i, S.shape[1] + i) for i in range(S.shape[1])}

	def generate_dual_problem(self, S, irrev, T, b, alt_ident=None):
		"""
		Args:
		  S:
		  irrev:
		  T:
		  b:
		"""
		m, n = S.shape
		Sxi, Sxr = S[:, irrev].T, np.delete(S, irrev, axis=1).T

		if alt_ident is not None:
			I = alt_ident
		else:
			I = np.identity(n)
		veclens = [("u", m), ("vp", I.shape[0]), ("vn", I.shape[0]), ("w", self.T.shape[0])]

		Ii, Ir = I[irrev, :], np.delete(I, irrev, axis=0)

		Ti, Tr = T[:, irrev].T, np.delete(T, irrev, axis=1).T

		var_prop_list = [[(pref + str(i), 0 if pref != "u" else None, None) for i in range(n)] for
						pref, n in
						veclens]


		Sdi = np.hstack([Sxi, Ii, -Ii, Ti, np.zeros([Sxi.shape[0], 1])])
		Sdr = np.hstack([Sxr, Ir, -Ir, Tr, np.zeros([Sxr.shape[0], 1])])
		Sd = np.vstack([Sdi, Sdr, np.zeros([1, Sdi.shape[1]])])
		names, v_lb, v_ub = list(zip(*chain(*var_prop_list)))

		names = list(names) + ['C']
		v_lb = list(v_lb) + [1]
		v_ub = list(v_ub) + [None]

		w_idx = [m + n + n + i for i in range(len(b))]
		Sd[-1, w_idx] = b
		Sd[-1, -1] = 1

		b_ub = np.hstack([np.array([None] * Sdi.shape[0]), np.array([0] * Sdr.shape[0]), np.array([0])])
		b_lb = np.array([0] * (Sd.shape[0]), dtype=object)
		b_ub[-1] = 0
		b_lb[-1] = None

		return Sd, VAR_CONTINUOUS, v_lb, v_ub, b_lb, b_ub, names


class BendersMasterSystem(GenericLinearSystem):
	def __init__(self, F, c, g, lb, ub, solver):
		"""
		Args:
		  F:
		  c:
		  g:
		  lb:
		  ub:
		  solver:
		"""
		var_names = ['x' + str(i) for i in range(F.shape[1])]
		super().__init__(S=F, var_types=VAR_INTEGER, lb=lb, ub=ub, b_lb=np.array([None] * F.shape[1]),
						 b_ub=g, var_names=var_names, solver=solver)
		self.c = c
		self.cuts = []

	def build_problem(self):
		super().build_problem()
		self.set_objective(self.c, True)

	def add_combinatorial_benders_cut(self, x_sol):
		"""
		Args:
		  x_sol:
		"""
		cut_coefs = x_sol.copy()
		pos_mask = cut_coefs > 1e-6
		cut_coefs[pos_mask] = -1
		cut_coefs[~pos_mask] = 1
		rhs = pos_mask.sum()
		self.cuts.extend(self.add_rows_to_model(cut_coefs.reshape(1, -1), [1 - rhs], [None]))

	def remove_cuts(self):
		self.remove_from_model(index=[c.name for c in self.cuts], what='const')


class BendersSlaveSystem(GenericLinearSystem):
	def __init__(self, A, M, D, b, e, lb_y, ub_y, solver=None):
		"""
		Args:
		  A:
		  M:
		  D:
		  b:
		  e:
		  lb_y:
		  ub_y:
		  solver:
		"""
		self.slack_vars = ['slack' + str(i) for i in range(A.shape[0])]
		self.y_var_names = ['y' + str(i) for i in range(A.shape[1])]
		var_names = self.y_var_names + self.slack_vars
		self.b_ub_fixed = np.array([None] * (A.shape[0] + D.shape[0]))
		self.lby, self.uby = np.array(lb_y), np.array(ub_y)
		Af = np.hstack([A, np.eye(A.shape[0])])
		Df = np.hstack([D, np.zeros([D.shape[0], A.shape[0]])])
		slack_bounds = np.zeros(A.shape[0])

		super().__init__(S=np.vstack([Af, Df]), var_types=VAR_CONTINUOUS,
						 lb=np.concatenate([self.lby, slack_bounds]),
						 ub=np.concatenate([self.uby, slack_bounds]),
						 b_lb=np.array([0] * (A.shape[0] + D.shape[0])), b_ub=self.b_ub_fixed,
						 var_names=var_names, solver=solver)

		self.e, self.M, self.b = e, M, b
		self.constraints = self.model.constraints
		self.y_var_mask = np.arange(0, A.shape[0])
		self.previous_changes = []

	def parametrize(self, x_sol):
		"""
		Args:
		  x_sol:
		"""
		if len(self.previous_changes) > 0:
			chgmap = zip(self.previous_changes, np.zeros(len(self.previous_changes)))
			self.model._set_variable_bounds_on_problem(chgmap, chgmap)

		rhs = np.concatenate([self.b - np.dot(self.M, x_sol.reshape(-1, 1)).ravel(), self.e.copy()], axis=0)
		nzi = np.nonzero(rhs)[0]
		self.previous_changes = [self.model.variables[self.slack_vars[k]] for k in nzi]
		values_to_change = rhs[nzi]
		nchgmap = zip(self.previous_changes, values_to_change)
		# self.set_constraint_bounds(self.constraints, lb=rhs, ub=self.b_ub_fixed)
		# lb_n, ub_n = np.concatenate([self.lby, -rhs]), np.concatenate([self.uby, -rhs])
		self.model._set_variable_bounds_on_problem(nchgmap, nchgmap)
