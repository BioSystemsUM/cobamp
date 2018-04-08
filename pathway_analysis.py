from optlang import Model, Variable, Constraint, Objective
import numpy as np
from itertools import chain

decompose_list = lambda a: chain.from_iterable(map(lambda i: i if isinstance(i, list) else [i], a))

class KShortestEnumerator(object):
	def __init__(self, linear_system):

		# Get linear system constraints and variables
		linear_system.build_problem()
		self.__ls_shape = linear_system.get_stoich_matrix_shape()
		self.model = linear_system.get_model()
		self.__dvars = linear_system.get_dvars()
		self.__c = linear_system.get_c_variable()

		# Open log files
		self.resf = open('results', 'w')
		self.logf = open('log', 'w')

		# Setup CPLEX parameters
		# TODO: Generalize parameter setup to other solvers
		self.__set_model_parameters()

		# Setup k-shortest constraints
		self.__add_kshortest_indicators()
		self.__add_exclusivity_constraints()
		self.__size_constraint = None
		self.__objective_expression = sum(decompose_list(self.__ivars))
		self.__set_objective()
		self.__integer_cuts = []
		self.__fix_indicators()
		self.set_size_constraint(1)
		self.__generate_var_map()
		self.model.update()

	def __fix_indicators(self):
		self.get_model()

	def __set_model_parameters(self):
		# self.model.problem.parameters.mip.tolerances.integrality.set(1e-9)
		# self.model.problem.parameters.feasopt.tolerance.set(1e-9)
		# self.model.problem.parameters.workmem.set(4096)
		# self.model.problem.parameters.clocktype.set(1)
		# self.model.problem.parameters.advance.set(0)
		# self.model.problem.parameters.mip.strategy.fpheur.set(1)
		# self.model.problem.parameters.emphasis.numerical.set(1)
		# self.model.problem.parameters.emphasis.mip.set(0)
		# self.model.problem.parameters.mip.display.set(5)
		# self.model.problem.parameters.simplex.tolerances.optimality.set(1e-9)
		# self.model.problem.parameters.mip.cuts.disjunctive.set(-1)
		# self.model.problem.set_results_stream(self.resf)
		# self.model.problem.set_log_stream(self.logf)
		pass

	def __add_kshortest_indicators(self):
		ivars = [[Variable(name=subvar.name + "_ind", type='binary') for subvar in var] if isinstance(var, list) else Variable(name=var.name + "_ind", type='binary') for var in self.__dvars]
		self.__ivars = ivars

		dvchain = list(decompose_list(self.__dvars))
		ivchain = list(decompose_list(self.__ivars))
		for var,ivar in zip(dvchain, ivchain):

			a, b, c, d = [Variable(ivar.name+name, type='binary') for name in ['a','b','c','d']]
			inactive = [Constraint(ivar + a, name='Inact1_'+ivar.name,lb=1, ub=1), Constraint(b - a, name='Inact2_'+ivar.name, lb=0),Constraint(var, name='Inact3_'+ivar.name, lb=0,ub=0, indicator_variable=b, active_when=1)]
			active = [Constraint(-ivar + c, name='Act1_'+ivar.name,lb=0, ub=0),Constraint(d - c, name='Act2_'+ivar.name, lb=0),Constraint(var - self.__c, name='Act3_'+ivar.name, lb=0, indicator_variable=d, active_when=1)]
			self.model.add(active)
			self.model.add(inactive)

	def __generate_var_map(self):
		self.__dvar_map, self.__ivar_map = [{i: [v.name for v in dv] if isinstance(dv, list) else dv.name for i,dv in enumerate(varset)} for varset in [self.__dvars, self.__ivars]]

	def __add_exclusivity_constraints(self):
		Cexclusive = [Constraint(sum(ivarpair), ub=1, name='_'.join([ivar.name for ivar in ivarpair])) for ivarpair in self.__ivars if isinstance(ivarpair, list)]
		self.model.add(Cexclusive)

	def __set_objective(self):
		objective = Objective(self.__objective_expression, direction='min')
		self.model.objective = objective

	def __get_ivar_sum_vector(self, value_map):
		return dict([[(svar.name for svar in var),sum(value_map[svar.name] for svar in var) if isinstance(var, list) else var.name, value_map[var.name]] for var in self.__ivars])

	def __integer_cut_count(self):
		return len(self.__integer_cuts)

	def __add_integer_cut(self, value_map, size):
		var_list = []
		for var_it in self.__ivars:
			if isinstance(var_it, list):
				if True in [value_map[var.name] == 1 for var in var_it]:
					var_list.extend(var_it)
			else:
				if value_map[var_it.name] == 1:
					var_list.append(var_it)
		constraint = Constraint(sum(var_list), name="IntCut"+str(self.__integer_cut_count()), ub=size-1)
		self.model.add(constraint)
		self.model.update()
		self.__integer_cuts.append(constraint)

	def set_size_constraint(self, start_at, end_at=None):
		if not self.__size_constraint is None:
			self.model.remove(self.__size_constraint)
		sc = Constraint(self.__objective_expression, name="SolSz", lb=start_at, ub=end_at)
		self.__size_constraint = sc
		self.model.add(sc)
		self.model.update()

	def get_ivar_names(self):
		return [var.name for var in chain(*self.__ivars)]

	def get_dvar_names(self):
		return [var.name for var in chain(*self.__dvars)]

	def get_model(self):
		return self.model

	def __optimize(self):
		'''
		:return: tuple (statusflag, Solution object)
		'''
		sol = status = None
		# try:
		status = self.model.optimize()
		if status == 'optimal':
			value_map = {v.name:v.primal for v in self.model.variables}
			sol = KShortestSolution(value_map, status, vmap=self.__dvar_map, imap=self.__ivar_map)
		# except Exception as e:
		# 	raise e
		return sol, status

	def get_single_solution(self):
		sol, status = self.__optimize()
		if sol is None:
			raise Exception('Solution is empty')
		self.__add_integer_cut(sol.var_values(), sol.attribute_value('size'))
		return sol, status

	def reset_enumerator_state(self):
		self.model.remove(self.__integer_cuts)
		self.__integer_cuts = []
		self.set_size_constraint(1)

	def log_close(self):
		self.resf.close()
		self.logf.close()


class LinearSystem(object):
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

		Ci = linear_constraints_from_matrix(S_full, vd, lb=0,ub=0, name="Ci")
		model = Model(name="linear_problem")

		model.add(Ci)

		self.__dvars = vi + list(map(list,zip(vrf, vrb)))
		self.__model = model

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

	def __set_attribute(self, key, value):
		self.__attribute_dict[key] = value

	def var_values(self):
		return self.__value_map

	def status(self):
		return self.__status

	def attribute_value(self, attribute_name):
		return self.__attribute_dict[attribute_name]

	def attribute_names(self):
		return self.__attribute_dict.keys()

class KShortestSolution(Solution):
	def __init__(self, value_map, status, vmap, imap):
		var_sum, indicator_sum = [{i: value_map[vs[0]] - value_map[vs[1]] if isinstance(vs, list) and type(vs) is not str else value_map[vs] for i, vs in mapd.items()} for mapd in [vmap, imap]]
		super().__init__(value_map, status, var_sum = var_sum, indicator_sum = indicator_sum, size = sum(0 not in v if isinstance(v,list) else v != 0 for v in indicator_sum.values()))

class DualLinearSystem(LinearSystem):
	def __init__(self, S, irrev, T, b):
		self.__model = None
		self.__ivars = None
		self.S, self.irrev, self.T, self.b = S, irrev, T, b
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
		# nRi, nRr = len(irrev), nR - len(irrev)
		veclens = [("u", nM), ("vp", nR), ("vn", nR), ("w", self.T.shape[1])]
		I = np.identity(nR)
		Sxi, Sxr = self.S[:, self.irrev].T, np.delete(self.S, self.irrev, axis=1).T
		Ii, Ir = I[self.irrev, :], np.delete(I, self.irrev, axis=0)
		Ti, Tr = self.T[:, self.irrev].T, np.delete(self.T, self.irrev, axis=1).T

		u, vp, vn, w = [[Variable(pref + str(i), lb=0 if pref != "u" else None) for i in range(n)] for pref, n in
						veclens]

		c = Variable(name="C", lb=1)
		self.__c = c
		Sdi = np.concatenate([Sxi, Ii, -Ii, Ti], axis=1)
		Sdr = np.concatenate([Sxr, Ir, -Ir, Tr], axis=1)

		vd = list(chain(u, vp, vn, w))

		Ci = linear_constraints_from_matrix(Sdi, vd, lb=0, name="Ci")
		Cr = linear_constraints_from_matrix(Sdr, vd, lb=0, ub=0, name="Cr")
		Cb = Constraint(sum([self.b[i] * w[i] for i in range(self.T.shape[0])]) + self.__c, lb=0,ub=0, name="Cb")

		model = Model(name="dual_problem")

		model.add(Ci)
		model.add(Cr)
		model.add(Cb)

		self.__dvars = list(map(list,zip(vp, vn)))
		self.__model = model

		return Sdi, Sdr

def linear_constraints_from_matrix(S, v, lb=None, ub=None, name="", verbose=False):
	cv = [Constraint(sum([S[i, j] * v[j] for j in range(S.shape[1])]), lb=lb, ub=ub, name=name + str(i)) for i in
		  range(S.shape[0])]
	if verbose:
		for c in cv:
			print(c)
	return cv

if __name__ == '__main__':
	S = np.array([[1, -1, 0, 0, -1, 0, -1, 0, 0],
				  [0, 1, -1, 0, 0, 0, 0, 0, 0],
				  [0, 1, 0, -1, -1, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 1, -1, 0, 0],
				  [0, 0, 0, 0, 0, 0, 1, -1, 0],
				  [0, 0, 0, 0, 1, 0, 0, 1, -1]])

	irrev = [0, 1, 2, 4, 5, 6, 7, 8]
	T = np.array([0] * S.shape[1]).reshape(1, S.shape[1])
	T[0, 8] = -1
	b = np.array([-1]).reshape(1, )

	system = DualLinearSystem(S, irrev, T, b)
	ksh = KShortestEnumerator(system)
	sols = [sol for sol in ksh.solution_iterator()]

	import pandas as pd
	vdf = pd.DataFrame({i: {k: v for k, v in sols[i].var_values().items()} for i in range(len(sols))})
	vdf.to_csv('test_sols.csv')

	print([np.where(sol.attribute_value('indicator_sum'))[0] for sol in sols])

	with open('test_model.txt', 'w') as f:
		f.write(ksh.model.to_lp())
	ksh.log_close()
