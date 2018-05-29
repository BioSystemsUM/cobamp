from optlang import Model, Variable, Constraint, Objective
from optlang.symbolics import Zero
import numpy as np
from itertools import chain
from multiprocessing import Pool, cpu_count


def get_linear_coefficients_from_vector(a, vars):
	return {vars[i]:a[i] for i in np.nonzero(a)[0]}

def set_constraint_from_vector(c, a, vars, verbose):
	coefx = get_linear_coefficients_from_vector(a, vars)
	c.set_linear_coefficients(coefx)
	if verbose:
		print(c)
	return c

def set_constraint_from_vector_fx(t):
	return set_constraint_from_vector(*t)

def linear_constraints_from_matrix(model, S, v, lb=None, ub=None, name="", verbose=False):
	M,N = S.shape
	constraint_list = [Constraint(Zero, lb=lb, ub=ub, name=name+str(i)) for i in range(M)]
	model.add(constraint_list)
	model.update()

	pool = Pool(cpu_count())
	pool_args = zip(constraint_list, S, [v]*M, [verbose]*M)
	pool.map(set_constraint_from_vector_fx, pool_args)

	return constraint_list


class IrreversibleLinearSystem(object):
	def __init__(self, S, irrev):
		self.__model = None
		self.__ivars = None
		self.S, self.irrev = S, irrev
		self.__c = None

	def get_model(self):
		return self.__model

	def get_stoich_matrix_shape(self):
		return self.S.shape

	def get_dvars(self):
		return self.__dvars

	def get_c_variable(self):
		return self.__c

	def build_problem(self):
		# Defining useful length constants
		nM, nR = self.S.shape
		nIrrev = len(self.irrev)
		nRev = nR - nIrrev
		veclens = [("irr", nIrrev), ("revfw", nRev), ("revbw", nRev)]
		Sxi, Sxr = self.S[:, self.irrev], np.delete(self.S, self.irrev, axis=1)

		vi, vrf, vrb = [[Variable(pref + str(i), lb=0) for i in range(n)] for pref, n in veclens]
		c = Variable(name="C", lb=1)
		self.__c = c

		S_full = np.concatenate([Sxi, Sxr, -Sxr], axis=1)

		vd = list(chain(vi, vrf, vrb))

		model = Model(name="linear_problem")
		Ci = linear_constraints_from_matrix(model, S_full, vd, lb=0, ub=0, name="Ci")

		model.add(Ci)

		self.__dvars = vi + list(map(list, zip(vrf, vrb)))
		self.__model = model

		return S_full


class LinearSystem(object):
	def __init__(self, S, lb, ub):
		self.__model = Model('LinearModel')
		self.S, self.lb, self.ub = S, lb, ub
		self.build_problem()

	def get_model(self):
		return self.__model

	def get_stoich_matrix_shape(self):
		return self.S.shape

	def build_problem(self):
		# Defining useful length constants
		nM, nR = self.S.shape
		self.v = [Variable('v' + i) for i in range(nR)]
		c = linear_constraints_from_matrix(self.S, self.v, self.lb, self.ub, "C")
		self.model.add(c)


class Solution(object):
	def __init__(self, value_map, status, **kwargs):
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
		self.__attribute_dict[key] = value

	def var_values(self):
		return self.__value_map

	def status(self):
		return self.__status

	def attribute_value(self, attribute_name):
		return self.__attribute_dict[attribute_name]

	def attribute_names(self):
		return self.__attribute_dict.keys()


class LinearSystemOptimizer(object):

	def __init__(self, lsystem):
		self.lsystem = lsystem
		self.__model = lsystem.get_model()

	def optimize(self, objective, minimize=False):
		self.__model.objective = Objective(
			sum([objective[i] * self.lsystem.v[i] for i in range(objective.shape[0])]))
		sol = status = None
		try:
			status = self.model.optimize()
			if status == 'optimal':
				value_map = {v.name: v.primal for v in self.model.variables}
				sol = Solution(value_map, status)
		except:
			pass
		return sol, status

	def var_range(self,i):
		minimum = self.optimize(np.array([i]), True)[0][vars[i].name]
		maximum = self.optimize(np.array([i]), False)[0][vars[i].name]
		return minimum, maximum
