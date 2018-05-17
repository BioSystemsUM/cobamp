from optlang import Model, Variable, Constraint, Objective
import cplex
from optlang.symbolics import Add, Zero
import numpy as np
from itertools import chain
from optimization import IrreversibleLinearSystem, Solution, linear_constraints_from_matrix

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
		self.model = linear_system.get_model()
		self.__dvars = linear_system.get_dvars()
		self.__c = linear_system.get_c_variable()
		self.__solzip = lambda x: zip(self.model.problem.variables.get_names(), x)

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
		self.__objective_expression = list(zip(list(self.__indicator_map.values()), [1]*len(self.__indicator_map.keys())))
		print(self.__objective_expression)
		self.__set_objective()
		self.__integer_cuts = []
		self.set_size_constraint(1)
		self.__current_size = 1
		self.__generate_var_map()
		self.model.update()

	def __set_model_parameters(self):
		self.model.parameters.mip.tolerances.integrality.set(1e-9)
		self.model.parameters.feasopt.tolerance.set(1e-9)
		self.model.parameters.workmem.set(4096)
		self.model.parameters.clocktype.set(1)
		self.model.parameters.advance.set(0)
		self.model.parameters.mip.strategy.fpheur.set(1)
		self.model.parameters.emphasis.numerical.set(1)
		self.model.parameters.emphasis.mip.set(0)
		self.model.parameters.mip.display.set(5)
		self.model.parameters.simplex.tolerances.optimality.set(1e-9)
		self.model.parameters.mip.cuts.disjunctive.set(-1)
		self.model.set_results_stream(self.resf)
		self.model.set_log_stream(self.logf)
		self.model.parameters.mip.limits.populate.set(999999)

	def __add_kshortest_indicators(self):
		"""

		:return:
		"""
		btype = self.model.variables.type.binary
		ivars = [[(subvar + "_ind", 0, 1, btype) for subvar in var] if isinstance(var, tuple) else (var + "_ind", 0, 1, btype) for var in self.__dvars]

		dvnames = list(chain(*list(decompose_list(self.__dvars))))
		ivchain = list(decompose_list(ivars))

		ivnames, ivlb, ivub, ivtype = list(zip(*ivchain))
		self.model.variables.add(names=ivnames, lb=ivlb, ub=ivub, types=''.join(ivtype))
		self.__ivars = ivnames
		print(dvnames)
		print(ivnames)

		self.__indicator_map = {}
		for var,ivar in zip(dvnames, ivnames):
			self.__indicator_map[var] = ivar
			auxvars =[(ivar+name, 0, 1, btype) for name in ['a','b','c','d']]
			auxname, auxlb, auxub, auxtype = list(zip(*auxvars))
			a,b,c,d = auxname
			self.model.variables.add(names=auxname, lb=auxlb, ub=auxub, types=auxtype)

			# auxiliary constraints
			aux_lin = [
				([ivar, a], [1, 1]),
				([a, b], [-1, 1]),
				([ivar, c], [-1, 1]),
				([c, d], [-1, 1])
			]
			aux_names = ['C'+ivar+'_helper'+str(i) for i in range(4)]
			self.model.linear_constraints.add(lin_expr=aux_lin, senses='EGEG', rhs=[1,0,0,0], names=aux_names)

			ind_lin = [([var], [1])]*2
			ind_names = ['C'+ivar+'_ind'+'1', 'C'+ivar+'_ind'+'2']
			print(ind_lin)
			self.model.indicator_constraints.add(lin_expr=ind_lin[0], sense='E', rhs=1, indvar=ivar, complemented=0, name=ind_names[0])
			self.model.indicator_constraints.add(lin_expr=ind_lin[1], sense='G', rhs=0, indvar=ivar, complemented=0, name=ind_names[1])

	def __generate_var_map(self):
		self.__dvar_map, self.__ivar_map = [{i: [v.name for v in dv] if isinstance(dv, list) else dv.name for i,dv in enumerate(varset)} for varset in [self.__dvars, self.__ivars]]

	def __add_exclusivity_constraints(self):
		lin_exprs = [([self.__indicator_map[var] for var in vlist],[1]*len(vlist)) for vlist in self.__dvars if isinstance(vlist, tuple)]
		nc = len(lin_exprs)
		self.model.linear_constraints.add(lin_exprs, senses='L'*nc, rhs=[1]*nc, names=['E'+str(i) for i in range(nc)])

	def __set_objective(self):
		self.model.objective.set_sense(self.model.objective.sense.minimize)
		self.model.objective.set_linear(self.__objective_expression)

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
		constraint = Constraint(Add(*var_list), name="IntCut"+str(self.__integer_cut_count()), ub=size-1)
	
		self.model.add(constraint)
		self.model.update()
		self.__integer_cuts.append(constraint)

	def set_size_constraint(self, start_at, end_at=None):
		# TODO: Find a way to add a single constraint with two bounds.
		if self.SIZE_CONSTRAINT_NAME in self.model.linear_constraints.get_names():

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
		:return: tuple (KShortestSolution sol)
		'''
		sol = None
		# try:
		status = self.model.optimize()
		if status == 'optimal':
			value_map = {v.name:v.primal for v in self.model.variables}
			sol = KShortestSolution(value_map, status, vmap=self.__dvar_map, imap=self.__ivar_map)
		return sol

	def __populate(self):

		sols = []
		first_sol = self.__optimize()
		if first_sol is not None:
			self.__add_integer_cut(first_sol.var_values(), first_sol.attribute_value('size'))
			sols.append(first_sol)
		else:
			print('Size constraint infeasible or solution space exhausted')
		self.model.problem.populate_solution_pool()
		n_sols = self.model.problem.solution.pool.get_num()
		for i in range(n_sols):
			value_map = dict(self.__solzip(self.model.problem.solution.pool.get_values(i)))
			sol = KShortestSolution(value_map, None, vmap=self.__dvar_map, imap=self.__ivar_map)
			sols.append(sol)
		for sol in sols:
			self.__add_integer_cut(sol.var_values(), sol.attribute_value('size'))

		return sols

	def enumeration_methods(self):
		return {self.ENUMERATION_METHOD_ITERATE: self.get_single_solution, self.ENUMERATION_METHOD_POPULATE: self.populate_next_size}
	
	def solution_iterator(self):
		self.reset_enumerator_state()
		self.set_size_constraint(1,None)
		failed = False
		while not failed:
			try:
				result = self.get_single_solution()
				yield result
			except Exception as e:
				print('Enumeration ended:',e)
				failed = False

	def population_iterator(self, max_size):
		self.reset_enumerator_state()
		for i in range(1, max_size+1):
			try:
				self.set_size_constraint(i, i)
				sols = self.populate_current_size()
				yield sols if sols is not None else []
			except:
				print('No solutions or error occurred at size ',i)

	def populate_current_size(self):
		sols = self.__populate()
		return sols

	def get_single_solution(self):
		sol = self.__optimize()
		if sol is None:
			raise Exception('Solution is empty')
		self.__add_integer_cut(sol.var_values(), sol.attribute_value('size'))
		return sol

	def reset_enumerator_state(self):
		self.model.remove(self.__integer_cuts)
		self.__integer_cuts = []
		self.set_size_constraint(1)

	def log_close(self):
		self.resf.close()
		self.logf.close()


class KShortestSolution(Solution):
	def __init__(self, value_map, status, vmap, imap):
		var_sum, indicator_sum = [{i: value_map[vs[0]] - value_map[vs[1]] if isinstance(vs, list) and type(vs) is not str else value_map[vs] for i, vs in mapd.items()} for mapd in [vmap, imap]]
		super().__init__(value_map, status, var_sum = var_sum, indicator_sum = indicator_sum, size = sum(0 not in v if isinstance(v,list) else v != 0 for v in indicator_sum.values()))

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

		u, vp, vn, w = [[(pref + str(i), 0 if pref != "u" else -CPLEX_INFINITY, CPLEX_INFINITY) for i in range(n)] for pref, n in
						veclens]

		Sdi = np.concatenate([Sxi, Ii, -Ii, Ti], axis=1)
		Sdr = np.concatenate([Sxr, Ir, -Ir, Tr], axis=1)
		Sd = np.concatenate([Sdi,Sdr], axis=0)
		vd = chain(u,vp,vn,w)
		names, lb, ub = list(zip(*vd))
		self.__model.variables.add(names=names, lb=lb, ub=ub)

		np_names = np.array(names)
		nnz = list(map(lambda y: np.nonzero(y)[1],zip(Sd)))
		print(nnz)

		lin_expr = [(np_names[x], row[x]) for row,x in zip(Sd, nnz)]
		rhs = [0] * (Sdi.shape[0] + Sdr.shape[0])
		senses = 'G' * Sdi.shape[0] + 'E' * Sdr.shape[0]
		cnames = ['Ci'+str(i) for i in range(Sdi.shape[0])] + ['Cr'+str(i) for i in range(Sdr.shape[0])]

		for row in lin_expr:
			print(row)
		self.__model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)

		self.__model.variables.add(names=['C'], lb=[1], ub=[CPLEX_INFINITY])

		b_coefs = self.b.tolist()+[1]
		b_names = list(list(zip(*w))[0] + tuple(['C']))
		print([(b_coefs,b_names)])
		self.__model.linear_constraints.add(lin_expr=[(b_names,b_coefs)], senses=['E'], rhs=[0], names=['Cb'])

		vp_names = list(zip(*vp))[0]
		vn_names = list(zip(*vn))[0]

		self.__dvars = list(zip(vp_names, vn_names))