import cplex, string, random, shutil

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
	folder = "tmp_"+random_string_generator(12)
	m_name, p_name = path.join(folder,random_string_generator(9)+".lp"), path.join(folder,random_string_generator(9)+".lp")
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

class LinearSystemOptimizer(object):
	"""
	Class with methods to solve a <LinearSystem> as a linear optimization problem.
	"""
	def __init__(self, linear_system):
		"""

		Parameters
		----------
		linear_system: A <LinearSystem> instance.
		"""
		linear_system.build_problem()
		self.model = copy_cplex_model(linear_system.get_model())
		self.model.set_results_stream(None)
		self.model.set_log_stream(None)

	def __optimize(self, objective, minimize):
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
		senses = self.model.objective.sense
		self.model.objective.set_linear(objective)
		self.model.objective.set_sense(senses.minimize if minimize else senses.maximize)

		try:
			self.model.solve()
			value_map = dict(zip(self.model.variables.get_names(),self.model.solution.get_values()))
			return Solution(value_map, self.model.solution.status)
		except Exception as e:
			print(e)

	def optimize(self, objective, minimize=False):
		"""
		See <self.__optimize>.
		Parameters
		----------
		objective
		minimize

		Returns
		-------

		"""
		return self.__optimize(objective, minimize)


