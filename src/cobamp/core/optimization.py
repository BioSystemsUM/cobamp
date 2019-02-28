import cplex, string, random, shutil
from numpy import nan, array, int_, float_
from optlang import Model, Variable, Constraint, Objective
from collections import OrderedDict

CPLEX_INFINITY = cplex.infinity


def random_string_generator(N):
	"""

	Parameters

	----------

		N : an integer

	Returns a random string of uppercase character and digits of length N
	-------

	"""
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))


def copy_cplex_model(model):
	"""
	Copies a Cplex model and returns a new object with the same contents in a separate object. Requires file creation
	and reading permissions.

	Parameters
	----------
		model : A cplex.Cplex instance

	Returns a deep copy of the model as a separate object.
	-------

	"""
	from os import mkdir, path
	folder = "tmp_" + random_string_generator(12)
	m_name, p_name = path.join(folder, random_string_generator(9) + ".lp"), path.join(folder, random_string_generator(
		9) + ".lp")
	mkdir(folder)

	model.write(m_name)
	model.parameters.write_file(p_name)

	new_model = cplex.Cplex()
	new_model.parameters.read_file(p_name)
	new_model.read(m_name)

	shutil.rmtree(folder)
	return new_model

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

		try:
			self.model.optimize()
			values = self.model._get_primal_values()
			value_map = OrderedDict([(k,v) for k,v in zip(names, values)])
			status = self.model.status
			ov = self.model.objective.value

		except Exception as e:
			frozen_exception = e

		if status or not self.hard_fail:
			return Solution(value_map, self.model.status, objective_value=ov)
		else:
			raise frozen_exception

class CORSOSolution(Solution):
	def __init__(self, sol, f, index_map, var_names):
		x = sol.x()
		nx = [x[i] if isinstance(x[i], float_) else x[i][0] - x[i][1] for i in range(max(index_map)+1)]
		nvalmap = OrderedDict([(k,v) for k,v in zip(var_names, nx)])
		super().__init__(nvalmap, sol.status(), objective_value=f)