import cplex, string, random, shutil
import numpy as np
from itertools import chain

CPLEX_INFINITY = cplex.infinity

def random_string_generator(N):
	return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))

def copy_cplex_model(model):
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

class IrreversibleLinearSystem(object):
	def __init__(self, S, irrev):
		self.__model = cplex.Cplex()
		self.__ivars = None
		self.S, self.irrev = S, irrev
		self.__c = "C"

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
		S_full = np.concatenate([Sxi, Sxr, -Sxr], axis=1)

		vi, vrf, vrb = [[(pref + str(i), 0, CPLEX_INFINITY) for i in range(n)] for pref, n in veclens]

		vd = chain(vi, vrf, vrb)
		names, lb, ub = list(zip(*vd))
		self.__model.variables.add(names=names, lb=lb, ub=ub)

		np_names = np.array(names)
		nnz = list(map(lambda y: np.nonzero(y)[1], zip(S_full)))

		lin_expr = [(np_names[x], row[x]) for row, x in zip(S_full, nnz)]

		rhs = [0] * S_full.shape[0]
		senses = 'E'* S_full.shape[0]
		cnames = ['C_' + str(i) for i in range(S_full.shape[0])]

		self.__model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)
		self.__model.variables.add(names=['C'], lb=[1], ub=[CPLEX_INFINITY])
		self.__dvars = vi + list(map(list, zip(vrf, vrb)))

		vi_names = list(zip(*vi))[0]
		vrf_names = list(zip(*vrf))[0]
		vrb_names = list(zip(*vrb))[0]

		self.__dvars = vi_names + list(zip(vrf_names,vrb_names))

		return S_full

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