import string, random, shutil
from numpy import nan, array, int_, float_, abs, zeros, max
import pandas as pd
from collections import OrderedDict


def random_string_generator(N):
	"""

	Parameters

	----------

		N : an integer

	Returns a random string of uppercase character and digits of length N
	-------

	"""
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


class Solution(object):
	"""
	Class representing a solution to a given linear optimization problem. Includes an internal dictionary for additional
	information to be included.
	"""

	def __init__(self, value_map, status, **kwargs):
		"""

		Parameters
		----------

			value_map: A dictionary mapping variable indexes with their values as determined by the solver

			status: An object (preferrably str or int) containing the solution status

			kwargs: Any additional information to be included in the attribute_dict variable that can be accessed by
			the <self.attribute_value> function.
		"""
		self.__value_map = value_map
		self.__status = status
		self.__attribute_dict = kwargs

		if 'objective_value' in kwargs:
			self.__obj_value = kwargs['objective_value']

	def __getitem__(self, item):
		if hasattr(item, '__iter__') and not isinstance(item, str):
			return {k: self.__value_map[k] for k in item}
		elif isinstance(item, str):
			return self.__value_map[item]
		else:
			raise TypeError('\'item\' is not a sequence or string.')

	def to_series(self):

		return pd.Series(self.var_values())

	def set_attribute(self, key, value):
		"""
		Sets the value of a given <key> as <value>.

		Parameters

		----------

			key - A string

			value - Any object to be associated with the supplied key
		-------

		"""
		self.__attribute_dict[key] = value

	def var_values(self):
		"""

		Returns a dict mapping reaction indices with the variable values.
		-------

		"""
		return self.__value_map

	def status(self):
		"""

		Returns the status of this solution, as supplied by the solver.
		-------

		"""
		return self.__status

	def attribute_value(self, attribute_name):
		"""

		Parameters

		----------

			attribute_name: A dictionary key (preferrably str)

		Returns the value associated with the supplied key
		-------

		"""
		return self.__attribute_dict[attribute_name]

	def attribute_names(self):
		"""

		Returns all keys present in the attribute dictionary.

		-------

		"""
		return self.__attribute_dict.keys()

	def objective_value(self):
		"""

		Returns the objective value for this solution

		"""
		return self.__obj_value

	def x(self):
		'''

		Returns a ndarray with the solution values in order (from the variables)

		'''

		return array(list(self.__value_map.values()))


class LinearSystemOptimizer(object):
	"""
	Class with methods to solve a <LinearSystem> as a linear optimization problem.
	"""

	def __init__(self, linear_system, hard_fail=False, build=True):
		"""

		Parameters

		----------

			linear_system: A <LinearSystem> instance.
			hard_fail: A boolean flag indicating whether an Exception is raised when the optimization fails
		"""
		if build:
			linear_system.build_problem()
		self.solver = linear_system.solver
		self.model = linear_system.get_model()
		self.hard_fail = hard_fail

	def optimize(self):
		"""
		Internal function to instantiate the solver and return a solution to the optimization problem

		Parameters

		----------

			objective: A List[Tuple[coef,name]], where coef is an objective coefficient and name is the name of the variable
			to be optimized.

			minimize: A boolean that, when True, defines the problem as a minimization

		Returns a <Solution> instance
		-------
		"""
		names = self.model._get_variables_names()

		value_map = OrderedDict([(v, nan) for v in names])
		status = None
		ov = nan

		# self.model.configuration.tolerances.feasibility = 1e-9 # TODO this is for a test, to delete later
		# self.model.configuration.tolerances.optimality = 1e-6 # TODO this is for a test, to delete later

		# tINIT test parameters
		# self.model.problem.Params.MIPGap = 1e-9
		# self.model.configuration.tolerances.feasibility = 1e-8
		# self.model.configuration.tolerances.optimality = 1e-8
		# self.model.configuration.verbosity = 3

		try:
			self.model.optimize()
			values = self.model._get_primal_values()
			value_map = OrderedDict([(k, v) for k, v in zip(names, values)])
			status = self.model.status
			ov = self.model.objective.value

		except Exception as e:
			frozen_exception = e

		if status or not self.hard_fail:
			return Solution(value_map, self.model.status, objective_value=ov)
		else:
			raise frozen_exception

	def populate(self, limit=None):
		intf_dict = {
			'CPLEX': self.__populate_cplex,
			'GUROBI': self.__populate_gurobi
		}
		if self.solver in ['CPLEX', 'GUROBI']:
			return intf_dict[self.solver](limit)
		else:
			raise ValueError('The provided solver does not have an implemented populate function. Choose from' +
							 ''.join(list(intf_dict.keys())))

	def __populate_cplex(self, limit=None):
		instance = self.model.problem

		if not limit:
			instance.parameters.mip.pool.capacity = instance.parameters.mip.pool.capacity.max()
		else:
			instance.parameters.mip.pool.capacity = limit
		vnames = instance.variables.get_names()
		mnames = self.model._get_variables_names()
		solutions = []
		try:
			instance.populate_solution_pool()
			pool_intf = instance.solution.pool
			nsols = pool_intf.get_num()
			for s in range(nsols):
				vmap = {k: v for k, v in zip(vnames, pool_intf.get_values(s))}
				ord_vmap = OrderedDict([(k, vmap[k]) for k in mnames])
				sol = Solution(ord_vmap, 'optimal', objective_value=pool_intf.get_objective_value(s))
				# TODO: get status dict from optlang and use it accordingly
				solutions.append(sol)
		except Exception as e:
			print(e)

		return solutions

	def __populate_gurobi(self, limit=None):

		instance = self.model.problem

		solutions = []
		instance.params.PoolSolutions = limit
		instance.params.SolutionNumber = 0
		try:
			instance.optimize()
			mnames = self.model._get_variables_names()
			if instance.SolCount > 0:
				for n in range(instance.SolCount):
					instance.params.SolutionNumber = n
					ord_vmap = OrderedDict([(k, instance.getVarByName(k).Xn) for k in mnames])
					sol = Solution(ord_vmap, 'optimal', objective_value=instance.PoolObjVal)
					solutions.append(sol)
		except Exception as e:
			print(e)
		finally:
			instance.params.SolutionNumber = 0

		return solutions


class CORSOSolution(Solution):
	def __init__(self, sol_max, sol_min, f, index_map, var_names, eps=1e-8):
		x = sol_min.x()
		rev = index_map[max(index_map) + 1:]

		nx = x[:max(index_map) + 1]
		nx[rev] = x[rev] - sol_min.x()[max(index_map) + 1:-1]
		nx[abs(nx) < eps] = 0
		nvalmap = OrderedDict([(k, v) for k, v in zip(var_names, nx)])
		super().__init__(nvalmap, [sol_max.status(), sol_min.status()], objective_value=f)


class GIMMESolution(Solution):
	def __init__(self, sol, exp_vector, var_names, mapping=None):
		self.exp_vector = exp_vector
		gimme_solution = sol.x()
		if mapping:
			gimme_solution = [max(gimme_solution[array(new)]) if isinstance(new, (tuple, list)) else gimme_solution[new]
							  for orig, new
							  in mapping.items()]
		super().__init__(
			value_map=OrderedDict([(k, v) for k, v in zip(var_names, gimme_solution)]),
			status=sol.status(),
			objective_value=sol.objective_value()
		)

	def get_reaction_activity(self, flux_threshold):
		gimme_fluxes = array([kv[1] for i, kv in enumerate(self.var_values().items())])
		activity = zeros(gimme_fluxes.shape)
		ones = (self.exp_vector > flux_threshold) | (self.exp_vector == -1)
		twos = gimme_fluxes > 0
		activity[ones] = 1
		activity[twos & ~ones] = 2

		return activity


class KShortestSolution(Solution):
	"""
	A Solution subclass that also contains attributes suitable for elementary flux modes such as non-cancellation sums
	of split reactions and reaction activity.
	"""
	SIGNED_INDICATOR_SUM = 'signed_indicator_map'
	SIGNED_VALUE_MAP = 'signed_value_map'

	def __init__(self, value_map, status, indicator_map, dvar_mapping, dvars, **kwargs):
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
			i: value_map[dvars[varlist[0]]] - value_map[dvars[varlist[1]]] if isinstance(varlist, (tuple, list)) else
			value_map[dvars[varlist]] for
			i, varlist in dvar_mapping.items()}
		signed_indicator_map = {
			i: value_map[indicator_map[dvars[varlist[0]]]] - value_map[indicator_map[dvars[varlist[1]]]] if isinstance(
				varlist,
				(tuple, list)) else
			value_map[indicator_map[dvars[varlist]]] for
			i, varlist in dvar_mapping.items()}
		super().__init__(value_map, status, **kwargs)
		self.set_attribute(self.SIGNED_VALUE_MAP, signed_value_map)
		self.set_attribute(self.SIGNED_INDICATOR_SUM, signed_indicator_map)

	def get_active_indicator_varids(self):
		"""

		Returns the indices of active indicator variables (maps with variables on the original stoichiometric matrix)

		-------

		"""
		return [k for k, v in self.attribute_value(self.SIGNED_INDICATOR_SUM).items() if abs(v) > 1e-10]
