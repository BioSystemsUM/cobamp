import cplex
import numpy as np
from itertools import chain
from optimization import IrreversibleLinearSystem, Solution, copy_cplex_model

CPLEX_INFINITY = cplex.infinity
decompose_list = lambda a: chain.from_iterable(map(lambda i: i if isinstance(i, list) else [i], a))


class KShortestEnumerator(object):
	ENUMERATION_METHOD_ITERATE = 'iterate'
	ENUMERATION_METHOD_POPULATE = 'populate'
	SIZE_CONSTRAINT_NAME = 'KShortestSizeConstraint'

	def __init__(self, linear_system):

		# Get linear system constraints and variables
		linear_system.build_problem()
		self.__ls_shape = linear_system.get_stoich_matrix_shape()
		self.model = copy_cplex_model(linear_system.get_model())
		self.__dvars = linear_system.get_dvars()
		self.__c = linear_system.get_c_variable()
		self.__solzip = lambda x: zip(self.model.variables.get_names(), x)

		# Open log files
		self.resf = open('results', 'w')
		self.logf = open('log', 'w')

		# Setup CPLEX parameters
		self.__set_model_parameters()

		# Setup k-shortest constraints
		self.__add_kshortest_indicators()
		self.__add_exclusivity_constraints()
		self.__size_constraint = None
		# TODO: change this to cplex notation
		self.__objective_expression = list(
			zip(list(self.__indicator_map.values()), [1] * len(self.__indicator_map.keys())))
		print(self.__objective_expression)
		self.__set_objective()
		self.__integer_cuts = []
		self.__exclusion_cuts = []
		self.set_size_constraint(1)
		self.__current_size = 1

	def __set_model_parameters(self):
		self.model.parameters.mip.tolerances.integrality.set(1e-9)
		self.model.parameters.workmem.set(4096)
		self.model.parameters.clocktype.set(1)
		self.model.parameters.advance.set(0)
		self.model.parameters.mip.strategy.fpheur.set(1)
		self.model.parameters.emphasis.mip.set(2)
		self.model.set_results_stream(self.resf)
		self.model.set_log_stream(self.logf)
		self.model.parameters.mip.limits.populate.set(100000)
		self.model.parameters.mip.pool.capacity.set(100000)
		self.model.parameters.mip.pool.intensity.set(4)
		self.model.parameters.mip.pool.absgap.set(0)
		self.model.parameters.mip.pool.replace.set(2)

	def exclude_solutions(self, sols):
		for sol in sols:
			if isinstance(sol, KShortestSolution):
				self.__add_integer_cut(sol.var_values())
			elif isinstance(sol, list) or isinstance(sol, tuple):
				ivars = [self.__indicator_map[k] for k in list(chain(*[self.__dvars[i] for i in sol]))]
				lin_expr = (ivars, [1]*len(ivars))
				sense = ['L']
				rhs = [len(sol)-1]
				names = ['exclusion_cuts'+str(len(self.__exclusion_cuts))]
				self.model.linear_constraints.add(lin_expr=[lin_expr], senses=sense, rhs=rhs, names=names)

	def __add_kshortest_indicators(self):
		"""

		:return:
		"""
		btype = self.model.variables.type.binary
		ivars = [[(subvar + "_ind", 0, 1, btype) for subvar in var] if isinstance(var, tuple) else (
		var + "_ind", 0, 1, btype) for var in self.__dvars]

		dvnames = []
		for elem in self.__dvars:
			if type(elem) is not str:
				dvnames = dvnames + list(elem)
			else:
				dvnames.append(elem)

		ivchain = list(decompose_list(ivars))

		ivnames, ivlb, ivub, ivtype = list(zip(*ivchain))
		self.model.variables.add(names=ivnames, lb=ivlb, ub=ivub, types=''.join(ivtype))
		self.__ivars = ivnames
		print(dvnames)
		print(ivnames)

		self.__indicator_map = {}
		for var, ivar in zip(dvnames, ivnames):
			self.__indicator_map[var] = ivar
			auxvars = [(ivar + name, 0, 1, btype) for name in ['a', 'b', 'c', 'd']]
			auxname, auxlb, auxub, auxtype = list(zip(*auxvars))
			a, b, c, d = auxname
			self.model.variables.add(names=auxname, lb=auxlb, ub=auxub, types=auxtype)

			# auxiliary constraints
			aux_lin = [
				([ivar, a], [1, 1]),
				([a, b], [-1, 1]),
				([ivar, c], [-1, 1]),
				([c, d], [-1, 1])
			]
			aux_names = ['C' + ivar + '_helper' + str(i) for i in range(4)]
			self.model.linear_constraints.add(lin_expr=aux_lin, senses='EGEG', rhs=[1, 0, 0, 0], names=aux_names)

			ind_lin = [([var], [1]),([var, 'C'],[1, -1])]
			ind_names = ['C' + ivar + '_ind' + '1', 'C' + ivar + '_ind' + '2']
			self.model.indicator_constraints.add(lin_expr=ind_lin[0], sense='E', rhs=0, indvar=b, complemented=0,
												 name=ind_names[0])
			self.model.indicator_constraints.add(lin_expr=ind_lin[1], sense='G', rhs=0, indvar=d, complemented=0,
												 name=ind_names[1])

	def __add_exclusivity_constraints(self):
		lin_exprs = [([self.__indicator_map[var] for var in vlist], [1] * len(vlist)) for vlist in self.__dvars if
					 isinstance(vlist, tuple)]
		nc = len(lin_exprs)
		self.model.linear_constraints.add(lin_exprs, senses='L' * nc, rhs=[1] * nc,
										  names=['E' + str(i) for i in range(nc)])

	def __set_objective(self):
		self.model.objective.set_sense(self.model.objective.sense.minimize)
		self.model.objective.set_linear(self.__objective_expression)

	def __get_ivar_sum_vector(self, value_map):
		return dict([[(svar.name for svar in var),
					  sum(value_map[svar.name] for svar in var) if isinstance(var, list) else var.name,
					  value_map[var.name]] for var in self.__ivars])

	def __integer_cut_count(self):
		return len(self.__integer_cuts)

	def __add_integer_cut(self, value_map):
		lin_expr_vars = []
		counter = 0
		for varlist in self.__dvars:
			if isinstance(varlist, tuple):
				if sum(abs(value_map[self.__indicator_map[var]]) for var in varlist) > 0:
					lin_expr_vars.extend([self.__indicator_map[var] for var in varlist])
					counter += 1
			else:
				if abs(value_map[self.__indicator_map[varlist]]) > 0:
					lin_expr_vars.append(self.__indicator_map[varlist])
					counter += 1

		self.model.linear_constraints.add(names=['cut' + str(len(self.__integer_cuts))],
									  lin_expr=[[lin_expr_vars, [1] * len(lin_expr_vars)]], senses=['L'],
									  rhs=[counter - 1])

	def set_size_constraint(self, start_at, equal=False):
		# TODO: Find a way to add a single constraint with two bounds.
		if self.SIZE_CONSTRAINT_NAME in self.model.linear_constraints.get_names():
			self.model.linear_constraints.set_rhs(self.SIZE_CONSTRAINT_NAME, start_at)
			self.model.linear_constraints.set_senses(self.SIZE_CONSTRAINT_NAME, 'E' if equal else 'G')
		else:
			lin_expr = [list(zip(*self.__objective_expression))]
			names = [self.SIZE_CONSTRAINT_NAME]
			senses = ['E' if equal else 'G']
			self.model.linear_constraints.add(lin_expr=lin_expr, names=names, senses=senses, rhs=[start_at])

	def __generate_var_map(self):
		self.__dvar_map, self.__ivar_map = [
			{i: [v.name for v in dv] if isinstance(dv, list) else dv.name for i, dv in enumerate(varset)} for varset in
			[self.__dvars, self.__ivars]]

	def get_ivar_names(self):
		return [var.name for var in chain(*self.__ivars)]

	def get_dvar_names(self):
		return [var.name for var in chain(*self.__dvars)]

	def get_model(self):
		return self.model

	def __optimize(self):
		'''
		:return: tuple (KShortestSolution sol)
		'''
		try:
			self.model.solve()
			status = self.model.solution.get_status()
			value_map = dict(zip(self.model.variables.get_names(), self.model.solution.get_values()))
			if status > -1:
				sol = KShortestSolution(value_map, status, self.__dvars, self.__indicator_map)
				return sol
		except Exception as e:
			print(e)


	def __populate(self):

		sols = []
		self.model.populate_solution_pool()
		n_sols = self.model.solution.pool.get_num()
		for i in range(n_sols):
			value_map = dict(self.__solzip(self.model.solution.pool.get_values(i)))
			sol = KShortestSolution(value_map, None, self.__dvars, self.__indicator_map)
			sols.append(sol)
		for sol in sols:
			self.__add_integer_cut(sol.var_values())
		return sols

	def enumeration_methods(self):
		return {self.ENUMERATION_METHOD_ITERATE: self.get_single_solution,
				self.ENUMERATION_METHOD_POPULATE: self.populate_next_size}

	def solution_iterator(self):
		self.reset_enumerator_state()
		self.set_size_constraint(1)
		failed = False
		while not failed:
			try:
				result = self.get_single_solution()
				yield result
			except Exception as e:
				print('Enumeration ended:', e)
				failed = True

	def population_iterator(self, max_size):
		self.reset_enumerator_state()
		for i in range(1, max_size + 1):
			try:
				self.set_size_constraint(i, True)
				sols = self.populate_current_size()
				yield sols if sols is not None else []
			except Exception as e:
				print('No solutions or error occurred at size ', i)
				raise e
	def populate_current_size(self):
		sols = self.__populate()
		return sols

	def get_single_solution(self):
		sol = self.__optimize()
		if sol is None:
			raise Exception('Solution is empty')
		self.__add_integer_cut(sol.var_values())
		return sol

	def reset_enumerator_state(self):
		self.model.linear_constraints.delete(self.__integer_cuts)
		self.set_size_constraint(1)


class KShortestSolution(Solution):
	def __init__(self, value_map, status, dvars, indicator_map, **kwargs):
		super().__init__(value_map,status,**kwargs)
		self.set_attribute('indicator_sum',self.indicators_from_value_map(value_map, dvars, indicator_map))
		self.set_attribute('signed_indicator_sum',self.signed_indicators_from_value_map(value_map, dvars, indicator_map))


	def indicators_from_value_map(self,value_map, dvars, indicator_map):
		isum = {
		i: sum(abs(value_map[indicator_map[var]]) for var in varlist) if isinstance(varlist, tuple) else abs(
			value_map[indicator_map[varlist]]) for i, varlist in enumerate(dvars)}
		return isum

	def signed_indicators_from_value_map(self,value_map, dvars, indicator_map):
		isum = {
		i: value_map[indicator_map[varlist[0]]] - value_map[indicator_map[varlist[1]]] if isinstance(varlist, tuple) else abs(
			value_map[indicator_map[varlist]]) for i, varlist in enumerate(dvars)}
		return isum


class DualLinearSystem(IrreversibleLinearSystem):
	def __init__(self, S, irrev, T, b):
		self.__model = cplex.Cplex()
		self.__ivars = None
		self.S, self.irrev, self.T, self.b = S, irrev, T, b
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
		# nRi, nRr = len(irrev), nR - len(irrev)
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
		self.__model.variables.add(names=names, lb=lb, ub=ub)

		np_names = np.array(names)
		nnz = list(map(lambda y: np.nonzero(y)[1], zip(Sd)))

		lin_expr = [cplex.SparsePair(ind=np_names[x].tolist(), val=row[x].tolist()) for row, x in zip(Sd, nnz)]
		rhs = [0] * (Sdi.shape[0] + Sdr.shape[0])
		senses = 'G' * Sdi.shape[0] + 'E' * Sdr.shape[0]
		cnames = ['Ci' + str(i) for i in range(Sdi.shape[0])] + ['Cr' + str(i) for i in range(Sdr.shape[0])]

		self.__model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)

		self.__model.variables.add(names=['C'], lb=[1], ub=[CPLEX_INFINITY])

		b_coefs = self.b.tolist() + [1]
		b_names = list(list(zip(*w))[0] + tuple(['C']))
		self.__model.linear_constraints.add(lin_expr=[(b_names, b_coefs)], senses=['L'], rhs=[0], names=['Cb'])

		vp_names = list(zip(*vp))[0]
		vn_names = list(zip(*vn))[0]

		self.__dvars = list(zip(vp_names, vn_names))
