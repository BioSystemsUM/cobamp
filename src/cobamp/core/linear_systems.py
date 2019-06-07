from collections import OrderedDict
from itertools import chain
import abc
import numpy as np
import warnings

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


def make_irreversible_model(S, lb, ub):
	irrev = np.array([i for i in range(S.shape[1]) if not (lb[i] < 0 and ub[i] > 0)])
	rev = np.array([i for i in range(S.shape[1]) if i not in irrev])
	Sr = S[:, rev]
	offset = S.shape[1]
	rx_mapping = {k: v if k in irrev else [v] for k, v in dict(zip(range(offset), range(offset))).items()}
	for i, rx in enumerate(rev):
		rx_mapping[rx].append(offset + i)
	rx_mapping = OrderedDict([(k, tuple(v)) if isinstance(v, list) else (k, v) for k, v in rx_mapping.items()])

	S_new = np.hstack([S, -Sr])
	nlb, nub = np.zeros(S_new.shape[1]), np.zeros(S_new.shape[1])
	for orig_rx, new_rx in rx_mapping.items():
		if isinstance(new_rx, tuple):
			nub[new_rx[0]] = abs(lb[orig_rx])
			nub[new_rx[1]] = ub[orig_rx]
		else:
			nlb[new_rx], nub[new_rx] = lb[orig_rx], ub[orig_rx]

	return S_new, nlb, nub, rx_mapping


class LinearSystem():
	"""
	An abstract class defining the template for subclasses implementing linear systems that can be used with optimizers
	such as CPLEX and passed onto other algorithms supplied with the package.

	Must instantiate the following variables:

		S: System of linear equations represented as a n-by-m ndarray, preferrably with dtype as float or int

		__model: Linear model as an instance of the solver.
	"""
	__metaclass__ = abc.ABCMeta
	model = None

	def select_solver(self, solver=None):
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

	@abc.abstractmethod
	def build_problem(self):
		"""
		Builds a CPLEX model with the constraints specified in the constructor arguments.
		This method must be implemented by any <LinearSystem>.
		Refer to the :ref:`constructor <self.__init__>`
		-------

		"""
		pass

	def get_model(self):
		"""

		Returns the model instance. Must call <self.build_problem()> to return a CPLEX model.

		-------

		"""
		return self.model

	def get_stoich_matrix_shape(self):
		"""

		Returns a tuple with the shape (rows, columns) of the supplied stoichiometric matrix.

		-------

		"""
		return self.S.shape

	# def empty_constraint(self, b_lb, b_ub, dummy_var=Variable(name='dummy', lb=0, ub=0)):
	def empty_constraint(self, b_lb, b_ub, **kwargs):
		return self.interface.Constraint(Zero, lb=b_lb, ub=b_ub, **kwargs)

	def dummy_variable(self):
		return self.interface.Variable(name='dummy', lb=0, ub=0)

	def populate_model_from_matrix(self, S, var_types, lb, ub, b_lb, b_ub, var_names, only_nonzero=False,
								   indicator_rows=None):
		'''

		Args:
			model: Optlang model
			S: Two-dimensional np.ndarray instance
			var_types: list or tuple with length equal to the amount of columns of S containing variable types (e.g.
			VAR_CONTINUOUS)
			lb: list or tuple with length equal to the amount of columns of S containing the lower bounds for all variables
			ub: list or tuple with length equal to the amount of columns of S containing the upper bounds for all variables
			b_lb: list or tuple with length equal to the amount of rows of S containing the lower bound for the right hand
			side of all constraints
			b_ub: list or tuple with length equal to the amount of rows of S containing the upper bound for the right hand
			side of all constraints
			var_names: string identifiers for the variables

		'''

		self.add_variables_to_model(var_names, lb, ub, var_types)
		self.add_rows_to_model(S, b_lb, b_ub, only_nonzero, indicator_rows)

	def populate_constraints_from_matrix(self, S, constraints, vars, only_nonzero=False):
		'''

		Args:
			S: Two-dimensional np.ndarray instance
			b_lb: list or tuple with length equal to the amount of rows of S containing the lower bound for the right hand
			side of all constraints
			b_ub: list or tuple with length equal to the amount of rows of S containing the upper bound for the right hand
			side of all constraints

			vars: list of variable instances

		'''
		if not only_nonzero:
			coef_list = [{vars[j]: S[i, j] for j in range(S.shape[1])} for i in range(S.shape[0])]
		else:
			coef_list = [{vars[j]: S[i, j] for j in np.nonzero(S[i, :])[0]} for i in range(S.shape[0])]

		for coefs, constraint in zip(coef_list, constraints):
			if constraint.indicator_variable is None:
				constraint.set_linear_coefficients(coefs)

		self.model.update()

	def add_rows_to_model(self, S_new, b_lb, b_ub, only_nonzero=False, indicator_rows=None, vars=None, names=None):
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
		vars = self.add_variables_to_model(var_names, lb, ub, var_types)

		self.populate_constraints_from_matrix(S_new, self.model.constraints, vars)

	def add_variables_to_model(self, var_names, lb, ub, var_types):

		if isinstance(var_types, str):
			var_types = [var_types] * len(lb)

		vars = [self.interface.Variable(name=name, lb=lbv, ub=ubv, type=t) for name, lbv, ubv, t in
				zip(var_names, lb, ub, var_types)]
		self.model.add(vars)
		# debug lines - TODO: remove this
		# for var in vars:
		# 	print(var)
		self.model.update()

		return vars

	def set_objective(self, coefficients, minimize, vars=None):
		if not vars:
			vars = self.model.variables
		lcoefs = {k: v for k, v in zip(vars, coefficients) if v != 0}
		dummy = self.dummy_variable()
		new_obj = self.interface.Objective(dummy)
		self.model.objective = new_obj
		new_obj.set_linear_coefficients(lcoefs)
		self.model.remove(dummy)
		self.model.objective.direction = SENSE_MINIMIZE if minimize else SENSE_MAXIMIZE
		self.model.update()

	def set_variable_bounds(self, vars, lb, ub):
		for var, ulb, uub in zip(vars, lb, ub):
			var.set_bounds(ulb, uub)
		self.model.update()

	def set_constraint_bounds(self, constraints, lb, ub):
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
		with open(filename, 'w') as f:
			f.write(self.model.to_lp())


class KShortestCompatibleLinearSystem(LinearSystem):
	"""
	Abstract class representing a linear system that can be passed as an argument for the KShortestAlgorithm class.
	Subclasses must instantiate the following variables:

		__dvar_mapping: A dictionary mapping reaction indexes with variable names

		__dvars: A list of variable names (str) or Tuple[str] if two linear system variables represent a single flux.
		Should be kept as type `list` to maintain order.

		__c: str representing the variable to be used as constant for indicator constraints
	"""
	dvar_mapping = None
	dvars = None
	c = None

	def get_dvar_mapping(self):
		"""

		Returns a dictionary mapping flux indices with variable(s) on the optimization problem
		-------

		"""
		return self.dvar_mapping

	def get_dvars(self):
		"""

		Returns a list of variables (str or Tuple[str]) with similar order to that of the fluxes passed to the system.
		-------

		"""
		return self.dvars

	def add_c_variable(self):
		self.c = self.add_variables_to_model(['C'], [1], [None], VAR_CONTINUOUS)[0]

	def get_c_variable(self):
		return self.model.variables['C']


class GenericLinearSystem(LinearSystem):
	"""
	Class representing a generic system of linear (in)equations
	Used as arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, var_types, lb, ub, b_lb, b_ub, var_names, solver=None):
		"""
		Constructor for GenericLinearSystem

		Parameters

			model: Optlang model
			S: Two-dimensional np.ndarray instance
			var_types: list or tuple with length equal to the amount of columns of S containing variable types (e.g.
			VAR_CONTINUOUS)
			lb: list or tuple with length equal to the amount of columns of S containing the lower bounds for all variables
			ub: list or tuple with length equal to the amount of columns of S containing the upper bounds for all variables
			b_lb: list or tuple with length equal to the amount of rows of S containing the lower bound for the right hand
			side of all constraints
			b_ub: list or tuple with length equal to the amount of rows of S containing the upper bound for the right hand
			side of all constraints
			var_names: string identifiers for the variables

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
	"""
	Class representing a steady-state biological system of metabolites and reactions without dynamic parameters
	Used as arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, lb, ub, var_names, solver=None):
		"""
		Constructor for SimpleLinearSystem

		Parameters

		----------

			S: Stoichiometric matrix represented as a n-by-m ndarray, preferrably with dtype as float or int

			lb: ndarray or list containing the lower bounds for all n fluxes

			ub: ndarray or list containing the lower bounds for all n fluxes

			var_names: - optional - ndarray or list containing the names for each flux
		"""
		m, n = S.shape
		self.lb, self.ub = lb, ub
		super().__init__(S, VAR_CONTINUOUS, lb, ub, [0] * m, [0] * m, var_names, solver=solver)


class IrreversibleLinearSystem(KShortestCompatibleLinearSystem, GenericLinearSystem):
	"""
	Class representing a steady-state biological system of metabolites and reactions without dynamic parameters.
	All irreversible reactions are split into their forward and backward components so every lower bound is 0.
	Used as arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, irrev, non_consumed=(), consumed=(), produced=(), solver=None):
		"""

		Parameters
		----------

			S: Stoichiometric matrix represented as a n-by-m ndarray, preferrably with dtype as float or int

			irrev: An Iterable[int] or ndarray containing the indices of irreversible reactions

			non_consumed: An Iterable[int] or ndarray containing the indices of external metabolites not consumed in the
			model.

			consumed: An Iterable[int] or ndarray containing the indices of external metabolites guaranteed to be produced.

			produced: An Iterable[int] or ndarray containing the indices of external metabolites guaranteed to be consumed.
		"""

		self.select_solver(solver)

		lb = [0 if i in irrev else -1 for i in range(S.shape[1])]
		ub = [1] * S.shape[1]
		Si, lbi, ubi, rev_mapping = make_irreversible_model(S, lb, ub)
		fwd_names = ['V' + str(i) if not isinstance(v, list) else 'V' + str(i) + 'F' for i, v in rev_mapping.items()]
		bwd_names = ['V' + str(i) + 'R' for i, v in rev_mapping.items() if isinstance(v, (list, tuple))]

		lbi = [0] * len(lbi) + [1]
		ubi = [None] * len(ubi) + [None]

		self.__ivars = None
		self.__ss_override = [(nc, 'G', 0) for nc in non_consumed] + [(p, 'G', 1) for p in produced] + [(c, 'L', -1) for
																										c in consumed]
		Si = np.hstack([Si, np.zeros((Si.shape[0], 1))])
		b_lb, b_ub = [0] * Si.shape[0], [0] * Si.shape[0]

		## TODO: Maybe allow other values to be provided for constraint relaxation/tightening

		for id, _, v in self.__ss_override:
			b_lb[id], b_ub[id] = (v, None) if v >= 0 else (None, v)

		super().__init__(Si, VAR_CONTINUOUS, lbi, ubi, b_lb, b_ub, fwd_names + bwd_names + ['C'], solver=solver)

		self.dvars = list(range(S.shape[1] + len(bwd_names)))
		self.dvar_mapping = dict(rev_mapping)


class IrreversibleLinearPatternSystem(IrreversibleLinearSystem):
	## TODO: Code + docstrings. Do not use this yet!
	def __init__(self, S, irrev, subset, **kwargs):
		super().__init__(S, irrev, **kwargs)
		self.subset = subset
		fwds, revs = [], []
		dvm_new = {}
		for r in self.subset:
			if r not in irrev:
				fid, rid = self.dvar_mapping[r]
				dvm_new[r] = (len(fwds),len(subset)+len(revs))
				fwds.append(fid)
				revs.append(rid)
			else:
				dvm_new[r] = len(fwds)
				fwds.append(self.dvar_mapping[r])

		self.dvar_mapping = dvm_new
		self.dvars = fwds + revs

	def build_problem(self):
		super().build_problem()


class DualLinearSystem(KShortestCompatibleLinearSystem, GenericLinearSystem):
	"""
	Class representing a dual system based on a steady-state metabolic network whose elementary flux modes are minimal
	cut sets for use with the KShortest algorithms. Based on previous work by Ballerstein et al. and Von Kamp et al.
	References:
	[1] von Kamp, A., & Klamt, S. (2014). Enumeration of smallest intervention strategies in genome-scale metabolic
	networks. PLoS computational biology, 10(1), e1003378.
	[2] Ballerstein, K., von Kamp, A., Klamt, S., & Haus, U. U. (2011). Minimal cut sets in a metabolic network are
	elementary modes in a dual network. Bioinformatics, 28(3), 381-387.
	"""

	def __init__(self, S, irrev, T, b, solver=None):
		"""

		Parameters

		----------

			S: Stoichiometric matrix represented as a n-by-m ndarray, preferrably with dtype as float or int
			irrev: An Iterable[int] or ndarray containing the indices of irreversible reactions

			T: Target matrix as an ndarray. Should have c-by-n dimensions (c - #constraints; n - #fluxes)

			b: Inhomogeneous bound values as a list or 1D ndarray of c size n.
		"""
		self.select_solver(solver)

		self.__ivars = None
		self.S, self.irrev, self.T, self.b = S, irrev, T, b
		self.__c = "C"
		super().__init__(*self.generate_dual_problem(S, irrev, T, b), solver=solver)

		offset = S.shape[0]
		self.dvars = list(range(offset, offset + (S.shape[1] * 2)))

		self.dvar_mapping = {i: (i, S.shape[1] + i) for i in range(S.shape[1])}

	def generate_dual_problem(self, S, irrev, T, b):
		m, n = S.shape
		veclens = [("u", m), ("vp", n), ("vn", n), ("w", self.T.shape[0])]
		I = np.identity(n)
		Sxi, Sxr = S[:, irrev].T, np.delete(S, irrev, axis=1).T
		Ii, Ir = I[irrev, :], np.delete(I, irrev, axis=0)
		Ti, Tr = T[:, irrev].T, np.delete(T, irrev, axis=1).T

		u, vp, vn, w = [[(pref + str(i), 0 if pref != "u" else None, None) for i in range(n)] for
						pref, n in
						veclens]
		Sdi = np.hstack([Sxi, Ii, -Ii, Ti, np.zeros([Sxi.shape[0], 1])])
		Sdr = np.hstack([Sxr, Ir, -Ir, Tr, np.zeros([Sxr.shape[0], 1])])
		Sd = np.vstack([Sdi, Sdr, np.zeros([1, Sdi.shape[1]])])
		names, v_lb, v_ub = list(zip(*list(chain(u, vp, vn, w))))

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
