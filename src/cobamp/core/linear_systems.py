from itertools import chain
import abc
import cplex
import numpy as np

from optlang import Model, Variable, Objective, Constraint
from cobamp.core.optimization import CPLEX_INFINITY


class LinearSystem():
	"""
	An abstract class defining the template for subclasses implementing linear systems that can be used with optimizers
	such as CPLEX and passed onto other algorithms supplied with the package.

	Must instantiate the following variables:

		S: System of linear equations represented as a n-by-m ndarray, preferrably with dtype as float or int

		__model: Linear model as an instance of the solver.
	"""
	__metaclass__ = abc.ABCMeta
	model = Model()
	S = None

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

	def empty_constraint(self, b_lb, b_ub, dummy_var=Variable(name='dummy', lb=0, ub=0)):
		return Constraint(dummy_var, lb=b_lb, ub=b_ub)

	def dummy_variable(self):
		return Variable(name='dummy', lb=0, ub=0)

	def populate_model_from_matrix(self, S, var_types, lb, ub, b_lb, b_ub, var_names):
		'''

		Args:
			model: Optlang model
			S: Two-dimensional numpy.ndarray instance
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
		self.add_rows_to_model(S, b_lb, b_ub)

	def populate_constraints_from_matrix(self, S, constraints, vars):
		'''

		Args:
			S: Two-dimensional numpy.ndarray instance
			b_lb: list or tuple with length equal to the amount of rows of S containing the lower bound for the right hand
			side of all constraints
			b_ub: list or tuple with length equal to the amount of rows of S containing the upper bound for the right hand
			side of all constraints

			vars: list of variable instances

		'''
		coef_list = [{vars[j]: S[i, j] for j in numpy.nonzero(S[i, :])[0]} for i in range(S.shape[0])]
		for coefs, constraint in zip(coef_list, constraints):
			constraint.set_linear_coefficients(coefs)

	def add_rows_to_model(self, S_new, b_lb, b_ub):
		vars = self.model.variables
		dummy = self.dummy_variable()
		constraints = [self.empty_constraint(b_lb[i], b_ub[i], dummy) for i in range(S_new.shape[0])]
		self.model.add(constraints)
		self.model.update()

		self.populate_constraints_from_matrix(S_new, constraints, vars)

		self.model.remove(dummy)
		self.model.update()

	def add_variables_to_model(self, var_names, lb, ub, var_types):

		if isinstance(var_types, str):
			var_types = [var_types] * len(lb)

		vars = [Variable(name=name, lb=lbv, ub=ubv, type=t) for name, lbv, ubv, t in zip(var_names, lb, ub, var_types)]
		self.model.add(vars)
		self.model.update()


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

	def get_c_variable(self):
		"""

		Returns a string referring to the constant on the optimization problem.
		-------

		"""
		return self.c


class GenericLinearSystem(LinearSystem):
	"""
	Class representing a generic system of linear (in)equations
	Used as arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, var_types, lb, ub, b_lb, b_ub, var_names):
		"""
		Constructor for GenericLinearSystem

		Parameters

			model: Optlang model
			S: Two-dimensional numpy.ndarray instance
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
		self.S, self.lb, self.ub, self.b_lb, self.b_ub, self.var_types = S, lb, ub, b_lb, b_ub, var_types

		self.names = var_names if var_names is not None else ['v' + str(i) for i in range(S.shape[1])]

	def build_problem(self):
		self.populate_model_from_matrix(self.S, self.var_types, self.lb, self.ub, self.b_lb, self.b_ub, self.names)


class SimpleLinearSystem(LinearSystem):
	"""
	Class representing a steady-state biological system of metabolites and reactions without dynamic parameters
	Used as arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, var_types, lb, ub, b_lb, b_ub, var_names):
		"""
		Constructor for SimpleLinearSystem

		Parameters

		----------

			S: Stoichiometric matrix represented as a n-by-m ndarray, preferrably with dtype as float or int

			lb: ndarray or list containing the lower bounds for all n fluxes

			ub: ndarray or list containing the lower bounds for all n fluxes

			var_names: - optional - ndarray or list containing the names for each flux
		"""
		self.S, self.lb, self.ub = S, lb, ub
		self.names = var_names if var_names is not None else ['v' + str(i) for i in range(S.shape[1])]

	def build_problem(self):
		np_names = np.array(self.names)
		nnz = list(map(lambda y: np.nonzero(y)[1], zip(self.S)))
		self.model.variables.add(names=self.names, lb=self.lb, ub=self.ub)
		lin_expr = [cplex.SparsePair(ind=np_names[x].tolist(), val=row[x].tolist()) for row, x in zip(self.S, nnz)]
		rhs = [0] * self.S.shape[0]
		senses = ['E'] * self.S.shape[0]
		cnames = ['C_' + str(i) for i in range(self.S.shape[0])]
		self.model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)


class IrreversibleLinearSystem(KShortestCompatibleLinearSystem):
	"""
	Class representing a steady-state biological system of metabolites and reactions without dynamic parameters.
	All irreversible reactions are split into their forward and backward components so every lower bound is 0.
	Used as arguments for various algorithms implemented in the package.
	"""

	def __init__(self, S, irrev, non_consumed=(), consumed=(), produced=()):
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
		self.model = cplex.Cplex()
		self.__ivars = None
		self.S, self.irrev = S, irrev
		self.__c = "C"
		self.__ss_override = [(nc, 'G', 0) for nc in non_consumed] + [(p, 'G', 1) for p in produced] + [(c, 'L', -1) for
																										c in consumed]

	def subset_dvars(self, subset):

		dvars = [self.dvar_mapping[i] for i in subset]
		dvar_mapping = {k: v for k, v in self.dvar_mapping.items() if v in dvars}
		return dvars, dvar_mapping

	def build_problem(self):
		# Defining useful length constants
		nM, nR = self.S.shape
		nIrrev = len(self.irrev)
		nRev = nR - nIrrev
		veclens = [("irr", nIrrev), ("revfw", nRev), ("revbw", nRev)]
		Sxi, Sxr = self.S[:, self.irrev], np.delete(self.S, self.irrev, axis=1)
		S_full = np.concatenate([Sxi, Sxr, -Sxr], axis=1)

		vi, vrf, vrb = [[(pref + str(i), 0, CPLEX_INFINITY) for i in range(n)] for pref, n in veclens]

		vd = chain(vi, vrf, vrb)
		names, lb, ub = list(zip(*vd))
		self.model.variables.add(names=names, lb=lb, ub=ub)

		np_names = np.array(names)
		nnz = list(map(lambda y: np.nonzero(y)[1], zip(S_full)))

		lin_expr = [cplex.SparsePair(ind=np_names[x].tolist(), val=row[x].tolist()) for row, x in zip(S_full, nnz)]

		rhs = [0] * S_full.shape[0]
		senses = ['E'] * S_full.shape[0]
		cnames = ['C_' + str(i) for i in range(S_full.shape[0])]

		if self.__ss_override != []:
			for id_i, sense_i, rhs_i in self.__ss_override:
				rhs[id_i] = rhs_i
				senses[id_i] = sense_i

		self.model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)
		self.model.variables.add(names=['C'], lb=[1], ub=[CPLEX_INFINITY])
		self.dvars = vi + list(map(list, zip(vrf, vrb)))

		vi_names = list(zip(*vi))[0] if len(vi) > 0 else []
		vrf_names = list(zip(*vrf))[0] if len(vrf) > 0 else []
		vrb_names = list(zip(*vrb))[0] if len(vrb) > 0 else []

		self.dvars = list(vi_names) + list(zip(vrf_names, vrb_names))
		var_index_sequence = (self.irrev.tolist() if isinstance(self.irrev, np.ndarray) else self.irrev) + [x for x in
																											list(range(
																												nR)) if
																											x not in self.irrev]

		self.dvar_mapping = dict(zip(var_index_sequence, self.dvars))
		# self.__model.write('efmmodel.lp') ## For debugging purposes
		return S_full


class IrreversibleLinearPatternSystem(IrreversibleLinearSystem):
	## TODO: Code + docstrings. Do not use this yet!
	def __init__(self, S, irrev, subset, **kwargs):
		super().__init__(S, irrev, **kwargs)
		self.subset = subset

	def build_problem(self):
		super().build_problem()
		self.dvars, self.dvar_mapping = self.subset_dvars(self.subset)
# model.


class DualLinearSystem(KShortestCompatibleLinearSystem):
	"""
	Class representing a dual system based on a steady-state metabolic network whose elementary flux modes are minimal
	cut sets for use with the KShortest algorithms. Based on previous work by Ballerstein et al. and Von Kamp et al.
	References:
	[1] von Kamp, A., & Klamt, S. (2014). Enumeration of smallest intervention strategies in genome-scale metabolic
	networks. PLoS computational biology, 10(1), e1003378.
	[2] Ballerstein, K., von Kamp, A., Klamt, S., & Haus, U. U. (2011). Minimal cut sets in a metabolic network are
	elementary modes in a dual network. Bioinformatics, 28(3), 381-387.
	"""

	def __init__(self, S, irrev, T, b):
		"""

		Parameters

		----------

			S: Stoichiometric matrix represented as a n-by-m ndarray, preferrably with dtype as float or int
			irrev: An Iterable[int] or ndarray containing the indices of irreversible reactions

			T: Target matrix as an ndarray. Should have c-by-n dimensions (c - #constraints; n - #fluxes)

			b: Inhomogeneous bound values as a list or 1D ndarray of c size n.
		"""
		self.model = cplex.Cplex()
		self.__ivars = None
		self.S, self.irrev, self.T, self.b = S, irrev, T, b
		self.__c = "C"

	def build_problem(self):
		# Defining useful length constants
		nM, nR = self.S.shape
		veclens = [("u", nM), ("vp", nR), ("vn", nR), ("w", self.T.shape[0])]
		I = np.identity(nR)
		Sxi, Sxr = self.S[:, self.irrev].T, np.delete(self.S, self.irrev, axis=1).T
		Ii, Ir = I[self.irrev, :], np.delete(I, self.irrev, axis=0)
		Ti, Tr = self.T[:, self.irrev].T, np.delete(self.T, self.irrev, axis=1).T

		u, vp, vn, w = [[(pref + str(i), 0 if pref != "u" else -CPLEX_INFINITY, CPLEX_INFINITY) for i in range(n)] for
						pref, n in
						veclens]

		Sdi = np.concatenate([Sxi, Ii, -Ii, Ti], axis=1)
		Sdr = np.concatenate([Sxr, Ir, -Ir, Tr], axis=1)
		Sd = np.concatenate([Sdi, Sdr], axis=0)
		vd = chain(u, vp, vn, w)
		names, lb, ub = list(zip(*vd))
		self.model.variables.add(names=names, lb=lb, ub=ub)

		np_names = np.array(names)
		nnz = list(map(lambda y: np.nonzero(y)[1], zip(Sd)))

		lin_expr = [cplex.SparsePair(ind=np_names[x].tolist(), val=row[x].tolist()) for row, x in zip(Sd, nnz)]
		rhs = [0] * (Sdi.shape[0] + Sdr.shape[0])
		senses = 'G' * Sdi.shape[0] + 'E' * Sdr.shape[0]
		cnames = ['Ci' + str(i) for i in range(Sdi.shape[0])] + ['Cr' + str(i) for i in range(Sdr.shape[0])]

		self.model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)

		self.model.variables.add(names=['C'], lb=[1], ub=[CPLEX_INFINITY])

		b_coefs = self.b.tolist() + [1]
		b_names = list(list(zip(*w))[0] + tuple(['C']))
		self.model.linear_constraints.add(lin_expr=[(b_names, b_coefs)], senses=['L'], rhs=[0], names=['Cb'])

		vp_names = list(zip(*vp))[0]
		vn_names = list(zip(*vn))[0]

		self.dvars = list(zip(vp_names, vn_names))
		self.dvar_mapping = dict(zip(range(len(self.dvars)), self.dvars))

		return Sd
