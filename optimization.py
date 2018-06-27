import cplex, string, random, shutil
import numpy as np
from itertools import chain

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