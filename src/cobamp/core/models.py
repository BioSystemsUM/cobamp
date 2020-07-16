import warnings
from collections import OrderedDict, Counter
from copy import deepcopy

from numpy import ndarray, array, delete, zeros, vstack, hstack, nonzero, append, int_, int8, int16, \
	int32, int64, where, isin

from cobamp.core.linear_systems import SteadyStateLinearSystem, VAR_CONTINUOUS, make_irreversible_model
from cobamp.utilities.context import CommandHistory
from cobamp.utilities.printing import pretty_table_print
from cobamp.core.optimization import LinearSystemOptimizer, CORSOSolution, GIMMESolution
from cobamp.core.cb_analysis import FluxVariabilityAnalysis
from cobamp.gpr.core import GPRContainer

INT_TYPES = (int, int_, int8, int16, int32, int64)
LARGE_NUMBER = 10e6 - 1
SMALL_NUMBER = 1e-6
BACKPREFIX = 'flux_backwards'

def to_list_if_single(x, n):
	if isinstance(x, (list, tuple)):
		return x
	else:
		return [x]*n

def make_irreversible_model_raven(S, lb, ub, inverse_reverse_reactions=False):
	irrev = array([i for i in range(S.shape[1]) if not (lb[i] < 0)])
	rev = array([i for i in range(S.shape[1]) if i not in irrev])
	Sr = S[:, rev]
	offset = S.shape[1]
	rx_mapping = {k: v if k in irrev else [v] for k, v in dict(zip(range(offset), range(offset))).items()}
	for i, rx in enumerate(rev):
		rx_mapping[rx].append(offset + i)
	rx_mapping = OrderedDict([(k, tuple(v)) if isinstance(v, list) else (k, v) for k, v in rx_mapping.items()])

	S_new = hstack([S, -Sr])
	nlb, nub = zeros(S_new.shape[1]), zeros(S_new.shape[1])
	for orig_rx, new_rx in rx_mapping.items():
		if isinstance(new_rx, tuple):
			nlb[new_rx[0]] = lb[orig_rx] if lb[orig_rx] >= 0 else 0  # not necessary since they're already 0
			nub[new_rx[0]] = ub[orig_rx] if ub[orig_rx] >= 0 else 0  # not necessary since they're already 0
			nlb[new_rx[1]] = ub[orig_rx] * -1 if (ub[orig_rx] * -1) >= 0 else 0
			nub[new_rx[1]] = lb[orig_rx] * -1 if (lb[orig_rx] * -1) >= 0 else 0

		else:
			nlb[new_rx] = lb[orig_rx]
			nub[new_rx] = ub[orig_rx]

	return S_new, nlb, nub, rx_mapping


# def make_irreversible_model(S, lb, ub):
# 	irrev = array([i for i in range(S.shape[1]) if not (lb[i] < 0 and ub[i] > 0)])
# 	rev = array([i for i in range(S.shape[1]) if i not in irrev])
# 	Sr = S[:, rev]
# 	offset = S.shape[1]
# 	rx_mapping = {k: v if k in irrev else [v] for k, v in dict(zip(range(offset), range(offset))).items()}
# 	for i, rx in enumerate(rev):
# 		rx_mapping[rx].append(offset + i)
# 	rx_mapping = OrderedDict([(k, tuple(v)) if isinstance(v, list) else (k, v) for k, v in rx_mapping.items()])
#
# 	S_new = hstack([S, -Sr])
# 	nlb, nub = zeros(S_new.shape[1]), zeros(S_new.shape[1])
# 	for orig_rx, new_rx in rx_mapping.items():
# 		if isinstance(new_rx, tuple):
# 			nub[new_rx[0]] = abs(lb[orig_rx])
# 			nub[new_rx[1]] = ub[orig_rx]
# 		else:
# 			nlb[new_rx], nub[new_rx] = lb[orig_rx], ub[orig_rx]
#
# 	return S_new, nlb, nub, rx_mapping


# def test_irreversible_conversions():
# 	S = zeros([3,3])
# 	lb, ub = ([0,1000],[-1000, 1000],[-1000, 0])


class ConstraintBasedModel(object):
	def __init__(self, S, thermodynamic_constraints, reaction_names=None, metabolite_names=None, optimizer=True,
				 solver=None, gprs=None):


		def __validate_args():

			# stoichiometric matrix
			if isinstance(S, ndarray):
				m, n = S.shape
			elif isinstance(S, tuple) or isinstance(S, list):
				m, n = len(S), len(S[0])
			else:
				raise TypeError('S matrix is not an ndarray or list')

			thermodynamic_types = set(map(type, thermodynamic_constraints))
			allowed_types = {list, tuple, bool}

			assert len(thermodynamic_types & allowed_types) == len(thermodynamic_types), 'Invalid bound type found.'

			reactions_ok = reaction_names is None or len(reaction_names) == n and isinstance(reaction_names, list)
			metabolites_ok = metabolite_names is None or len(metabolite_names) == m and isinstance(metabolite_names,
																								   list)

			assert reactions_ok, 'Reaction name dimensions do not match the supplied stoichiometrix matrix'
			assert metabolites_ok, 'Metabolite name dimensions do not match the supplied stoichiometrix matrix'
			return m, n

		self.solver = solver
		m, n = __validate_args()
		# self.__S = self.__parse_stoichiometric_matrix(S)
		self.__S = array(S)

		self.bounds = self.__interpret_bounds(thermodynamic_constraints)
		self.original_bounds = deepcopy(self.bounds)
		self.reaction_names, self.metabolite_names = deepcopy(reaction_names), deepcopy(metabolite_names)

		self.__update_decoder_map()

		if gprs is not None:
			self.gpr = gprs
		else:
			self.gpr = ['']*len(self.reaction_names)
		self.c = None
		self.model = None

		if optimizer:
			self.initialize_optimizer()

		self.context_managers = []

	def has_context(self):
		return len(self.context_managers) > 0

	def __enter__(self):
		self.context_managers.append(CommandHistory())
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		ctx_man = self.context_managers.pop()
		ctx_man.execute_all(False)

	def __getstate__(self):
		return self.__dict__

	def __update_decoder_map(self):
		self.reaction_decoder_map = self.metabolite_decoder_map = None

		if self.reaction_names:
			self.reaction_decoder_map = dict(zip(self.reaction_names, list(range(self.__S.shape[1]))))
		if self.metabolite_names:
			self.metabolite_decoder_map = dict(zip(self.metabolite_names, list(range(self.__S.shape[0]))))

		self.map_labels = {'reaction': self.reaction_decoder_map,
						   'metabolite': self.metabolite_decoder_map}

	def __interpret_bounds(self, thermodynamic_constraints):
		def interpret_bound(bound_obj):
			i, bound = bound_obj
			#			value = [None, None]
			if isinstance(bound, bool):
				if bound:
					value = 0, LARGE_NUMBER
				else:
					value = -LARGE_NUMBER, LARGE_NUMBER
			else:
				value = bound
				for i, v in enumerate(value):
					if i == 0 and v is None:
						value[i] = -LARGE_NUMBER
					elif i == 1 and v is None:
						value[i] = LARGE_NUMBER
			if value[0] > value[1]:
				warnings.warn('Lower bound is >= than upper bound for reaction ' + str(i) + '. Reversing...')
				value = value[::-1]
			return value

		return list(map(interpret_bound, enumerate(thermodynamic_constraints)))

	def flux_limits(self, reaction):
		self.set_objective({reaction: 1}, True)
		min_flux = self.optimize().objective_value()
		self.set_objective({reaction: 1}, False)
		max_flux = self.optimize().objective_value()

		return min_flux, max_flux

	def simplify(self, initial_objective, objective_sense, fva_gamma=1-1e-6):
		res = FluxVariabilityAnalysis(self.model).run(
			initial_objective=initial_objective, minimize_initial=objective_sense, gamma=fva_gamma)
		blocked_reaction_names = [self.reaction_names[i] for i in res.find_blocked_reactions()]
		self.remove_reactions(blocked_reaction_names)
		self.remove_orphan_metabolites()


	@property
	def gpr(self):
		return self.__gpr

	@gpr.setter
	def gpr(self, gprs, **kwargs):
		if isinstance(gprs, (list, tuple)):
			assert not (False in [isinstance(gpr, str) for gpr in gprs]), 'Non-string object found in gprs iterable'
			self.__gpr = GPRContainer(gpr_list=gprs, **kwargs)
		elif isinstance(gprs, GPRContainer):
			assert len(gprs) == len(self.bounds)
			self.__gpr = gprs
		else:
			raise(TypeError,'Can\'t use '+str(type(gprs))+' to redefine this model\'s GPRs')

	def remove_orphan_metabolites(self):
		zero_rows = [self.metabolite_names[i] for i in where((self.get_stoichiometric_matrix() == 0).all(axis=1))[0]]
		if len(zero_rows) > 0:
			self.remove_metabolites(zero_rows)

	def remove_orphan_reactions(self):
		zero_cols = [self.metabolite_names[i] for i in where((self.get_stoichiometric_matrix() == 0).all(axis=0))[0]]
		if len(zero_cols) > 0:
			self.remove_reactions(zero_cols)

	def initialize_optimizer(self):
		lb, ub = self.get_bounds_as_list()
		self.model = SteadyStateLinearSystem(self.__S, lb, ub, var_names=self.reaction_names, solver=self.solver)
		self.model.build_problem()
		self.optimizer = LinearSystemOptimizer(self.model, build=False)

	def decode_index(self, index, labels):
		if type(index) in INT_TYPES:
			return index

		elif labels is not None:
			return self.map_labels[labels][index]
		else:
			raise IndexError('Could not find specified index "' + str(index) + '".')

	def get_bounds_as_list(self):
		return list(zip(*self.bounds))

	def is_reversible_reaction(self, index):
		lb, ub = self.bounds[self.decode_index(index, 'reaction')]
		return (lb < 0 and ub > 0)

	def get_stoichiometric_matrix(self, rows=None, columns=None):
		row_index = [self.decode_index(i, 'metabolite') for i in rows] if rows else None
		col_index = [self.decode_index(i, 'reaction') for i in columns] if columns else None

		if rows and columns:
			return self.__S[row_index, col_index]
		elif rows:
			return self.__S[row_index, :]
		elif columns:
			return self.__S[:, col_index]
		else:
			return self.__S

	def set_stoichiometric_matrix(self, values, rows=None, columns=None, update_only_nonzero=False):
		row_index = [self.decode_index(i, 'metabolite') for i in rows] if rows else None
		col_index = [self.decode_index(i, 'reaction') for i in columns] if columns else None

		if self.has_context():
			self.context_managers[-1].queue_command(
				self.set_stoichiometric_matrix,
				{
					'values': self.get_stoichiometric_matrix(rows, columns),
					'rows': rows,
					'columns': columns
				}
			)

		if rows and columns:
			self.__S[row_index, col_index] = values
		elif rows:
			self.__S[row_index, :] = values
		elif columns:
			self.__S[:, col_index] = values
		else:
			self.__S = values

		row_index = row_index if row_index is not None else range(self.__S.shape[0])
		col_index = col_index if col_index is not None else range(self.__S.shape[1])

		if self.model:
			constraints = [self.model.model.constraints[i] for i in
						   row_index] if row_index else self.model.model.constraints
			vars = [self.model.model.variables[i] for i in col_index] if row_index else self.model.model.variables

			self.model.populate_constraints_from_matrix(values, constraints=constraints,
			                                            vars=vars, only_nonzero=update_only_nonzero)

	def add_metabolites(self, args, names=None):
		assert sum([n in self.reaction_names for n in names]) == 0, 'Duplicate metabolite name found!'
		if isinstance(args, list):
			rows = zeros(len(args), self.__S.shape[1])
			for row_i in range(len(args)):
				for k, v in args[row_i].items():
					rows[row_i,self.decode_index(k, 'reaction')] = v
		elif isinstance(args, ndarray):
			if args.shape[1] == len(self.reaction_names):
				rows = args
			else:
				raise Exception('Numpy argument dimensions should be (', len(self.metabolite_names), ',). Got',
								args.shape, 'instead')
		else:
			raise ValueError('Invalid argument type: ', type(args), '. Please supply an ndarray or dict instead.')

		if self.has_context():
			self.context_managers[-1].queue_command(self.remove_metabolites, self.__S.shape[0])

		self.__S = vstack([self.__S, rows])
		self.metabolite_names.extend(names)

		self.__update_decoder_map()

		if self.model:
			self.model.add_rows_to_model(rows.reshape([1, self.__S.shape[1]])
			                             if rows.size == self.__S.shape[1] else rows,
										 b_lb=array([0]*rows.shape[0]), b_ub=array([0]*rows.shape[0]),
										 only_nonzero=True)

	def add_reactions(self, args, bounds, names=None, gpr=None):
		assert sum([n in self.reaction_names for n in names]) == 0, 'Duplicate reaction name found!'
		if isinstance(args, (list,tuple)):
			cols = zeros([self.__S.shape[0], len(args)])
			for col_i in range(len(args)):
				for k, v in args[col_i].items():
					cols[self.decode_index(k, 'metabolite'), col_i] = v
		elif isinstance(args, ndarray):
			if args.shape[0] == len(self.metabolite_names):
				cols = args
			else:
				raise Exception('Numpy argument dimensions should be (','m',',', len(self.reaction_names),'). Got',
								str(args.shape), 'instead')
		else:
			raise ValueError('Invalid argument type: ', type(args), '. Please supply an ndarray or dict instead.')

		if self.has_context():
			self.context_managers[-1].queue_command(self.remove_reactions,
													{'index': [self.__S.shape[1]+i for i in range(cols.shape[1])]})

		self.__S = hstack([self.__S, cols])

		self.reaction_names.extend(names)
		self.bounds.extend(bounds)

		self.__update_decoder_map()
		if gpr is None:
			gprs = ['']*cols.shape[1]
		else:
			gprs = gpr
		self.gpr.add_gprs(gprs)

		lbs, ubs = zip(*bounds)
		if self.model:
			self.model.add_columns_to_model(cols, names, lbs, ubs, VAR_CONTINUOUS, only_nonzero=True)

	def add_metabolite(self, arg, name=None):
		assert not name in self.metabolite_names, 'Duplicate metabolite name found!'
		if isinstance(arg, dict):
			row = zeros(1, self.__S.shape[1])
			for k, v in arg.items():
				row[self.decode_index(k, 'reaction')] = v
		elif isinstance(arg, ndarray):
			if len(arg) == len(self.reaction_names):
				row = arg
			else:
				raise Exception('Numpy argument dimensions should be (', len(self.metabolite_names), ',). Got',
								arg.shape, 'instead')
		else:
			raise ValueError('Invalid argument type: ', type(arg), '. Please supply an ndarray or dict instead.')

		if self.has_context():
			self.context_managers[-1].queue_command(self.remove_metabolites,self.__S.shape[0])

		self.__S = vstack([self.__S, row])
		self.metabolite_names.append(name)

		self.__update_decoder_map()

		if self.model:
			self.model.add_rows_to_model(row.reshape([1, self.__S.shape[1]]), b_lb=array([0]), b_ub=array([0]), only_nonzero=True)

	def add_reaction(self, arg, bounds, name=None, gpr=''):
		assert not name in self.reaction_names, 'Duplicate reaction name found!'
		if isinstance(arg, dict):
			col = zeros([self.__S.shape[0], 1])
			for k, v in arg.items():
				col[self.decode_index(k, 'metabolite'), 0] = v
		elif isinstance(arg, ndarray):
			if len(arg) == len(self.metabolite_names):
				col = arg.reshape(self.__S.shape[0], 1)
			else:
				raise Exception('Numpy argument dimensions should be (', len(self.reaction_names), ',). Got', arg.shape,
								'instead')
		else:
			raise ValueError('Invalid argument type: ', type(arg), '. Please supply an ndarray or dict instead.')

		if self.has_context():
			self.context_managers[-1].queue_command(self.remove_reactions,{'index':self.__S.shape[1]})

		self.__S = hstack([self.__S, col])

		self.reaction_names.append(name)
		self.bounds.append(bounds)

		self.__update_decoder_map()
		self.gpr.add_gprs([gpr])

		if self.model:
			self.model.add_columns_to_model(col, [name], [bounds[0]], [bounds[1]], VAR_CONTINUOUS, only_nonzero=True)

	def remove_reaction(self, index):
		warnings.warn('''remove_reaction will be renamed to remove_reactions in a future version to represent the
					  method\'s true behaviour''',DeprecationWarning)
		self.remove_reactions(index)

	def remove_metabolite(self, index):
		warnings.warn('''remove_metabolite will be renamed to remove_metabolites in a future version to represent the
					  method\'s true behaviour''',DeprecationWarning)
		self.remove_metabolites(index)

	def get_boundary_reactions(self, epsilon=1e-9):
		nzcoefs = where(abs(self.get_stoichiometric_matrix()) > epsilon)
		boundary_rx_ids = [k for k,v in Counter(nzcoefs[1]).items() if v == 1]
		x = array(nzcoefs)
		boundaries = {}
		for k,v in zip(*x[:,isin(x[1], boundary_rx_ids)]):
			ki, vi = self.metabolite_names[k], self.reaction_names[v]
			if ki not in boundaries:
				boundaries[ki] = [vi]
			else:
				boundaries[ki].append(vi)
		return boundaries

	def add_boundary_reactions(self, metabolites, lbs, ubs, prefix='EX_', epsilon=1e-9):
		lbs, ubs = (to_list_if_single(x, len(metabolites)) for x in [lbs, ubs])

		return self.add_reactions(args=[{k:-1} for k in metabolites],
		                   bounds=[(l,u) for l,u in zip(lbs, ubs)],
		                   names=[prefix+m for m in metabolites],
		                   gpr=['' for m in metabolites])

	def remove_reactions(self, index):
		if isinstance(index, (int, str)):
			index = [index]

		j = [self.decode_index(i, 'reaction') for i in index]

		if self.has_context():
			for decoded_index in j:
				self.context_managers[-1].queue_command(
					self.add_reaction,
					{
						'arg':self.get_stoichiometric_matrix(rows=None, columns=[decoded_index]),
						'name':self.reaction_names[decoded_index],
						'bounds':self.get_reaction_bounds(decoded_index),
						'gpr':self.gpr[decoded_index]
					}
				)


		if self.reaction_names is not None:
			new_rnames = [self.reaction_names[k] for k in range(len(self.reaction_names)) if k not in j]
			self.reaction_names = new_rnames

		new_bounds = [k for i,k in enumerate(self.bounds) if i not in j]
		self.bounds = new_bounds
		self.__S = delete(self.__S, j, axis=1)
		self.__update_decoder_map()

		self.gpr.remove_gprs(index)

		if self.model:
			self.model.remove_from_model(j, 'var')

	def remove_metabolites(self, index):
		if isinstance(index, (int, str)):
			index = [index]

		i = [self.decode_index(ii, 'metabolite') for ii in index]

		if self.has_context():
			for decoded_index in i:
				self.context_managers[-1].queue_command(
					self.add_metabolite,
					{
						'arg':self.get_stoichiometric_matrix(rows=[decoded_index], columns=None).ravel(),
						'name':self.metabolite_names[decoded_index],
					}
				)


		if self.reaction_names is not None:
			new_mnames = [k for i1,k in enumerate(self.metabolite_names) if i1 not in i]
			self.metabolite_names = new_mnames

		self.__S = delete(self.__S, i, axis=0)
		self.__update_decoder_map()

		if self.model:
			self.model.remove_from_model(i, 'const')

	def make_irreversible(self):
		lb, ub = self.get_bounds_as_list()
		Sn, nlb, nub, rx_mapping = make_irreversible_model(self.__S, lb, ub)
		rev = array([int(v[0]) for k, v in rx_mapping.items() if isinstance(v, (list, tuple))])

		gprs_irrev = ['']*len(nlb)
		for i,tup in rx_mapping.items():
			if isinstance(tup, (tuple, list)):
				gprs_irrev[tup[0]] = gprs_irrev[tup[1]] = self.gpr[i]
			else:
				gprs_irrev[tup] = self.gpr[i]

		if self.reaction_names != None:
			if len(rev) > 0:
				revnames = ['_'.join([name, BACKPREFIX]) for name in (array(self.reaction_names)[rev]).tolist()]
			else:
				revnames = []
			rnames = self.reaction_names + revnames

			model = ConstraintBasedModel(Sn, list(zip(nlb, nub)), rnames, self.metabolite_names,
										 optimizer=self.model is not None, solver=self.solver, gprs=gprs_irrev)
		else:
			model = ConstraintBasedModel(Sn, list(zip(nlb, nub)), optimizer=self.model is not None, solver=self.solver,
										 gprs=gprs_irrev)
		return model, rx_mapping

	def get_reaction_bounds(self, index):
		return self.bounds[self.decode_index(index, 'reaction')]

	def set_reaction_bounds(self, index, **kwargs):
		true_idx = self.decode_index(index, 'reaction')
		lb, ub = self.get_reaction_bounds(true_idx)

		if self.has_context():
			self.context_managers[-1].queue_command(
				self.set_reaction_bounds,
				{
					'index':true_idx,
					'lb': lb,
					'ub': ub
				}
			)

		if 'lb' in kwargs:
			lb = kwargs['lb']
		if 'ub' in kwargs:
			ub = kwargs['ub']
		bound = (lb, ub)
		self.bounds[true_idx] = bound

		if 'temporary' in kwargs and kwargs['temporary'] == False:
			self.original_bounds[self.decode_index(index, 'reaction')] = self.get_reaction_bounds(index)

		if self.model:
			var = self.model.model.variables[true_idx]
			self.model.set_variable_bounds([var], [lb], [ub])

	# def set_reaction_bounds(self, indices, lb, ub, temporary=False):
	# 	true_idx = self.decode_index(index, 'reaction')
	# 	lb, ub = self.get_reaction_bounds(true_idx)
	#
	# 	if self.has_context():
	# 		self.context_managers[-1].queue_command(
	# 			self.set_reaction_bounds,
	# 			{
	# 				'index':true_idx,
	# 				'lb': lb,
	# 				'ub': ub
	# 			}
	# 		)
	#
	# 	if 'lb' in kwargs:
	# 		lb = kwargs['lb']
	# 	if 'ub' in kwargs:
	# 		ub = kwargs['ub']
	# 	bound = (lb, ub)
	# 	self.bounds[true_idx] = bound
	#
	# 	if 'temporary' in kwargs and kwargs['temporary'] == False:
	# 		self.original_bounds[self.decode_index(index, 'reaction')] = self.get_reaction_bounds(index)
	#
	# 	if self.model:
	# 		var = self.model.model.variables[true_idx]
	# 		self.model.set_variable_bounds([var], [lb], [ub])


	def set_objective(self, coef_dict, minimize=False):
		if self.model:
			if isinstance(coef_dict, dict):
				f = zeros(self.__S.shape[1], )
				self.c = f
				for k, v in coef_dict.items():
					self.c[self.decode_index(k, 'reaction')] = v
				self.model.set_objective(self.c, minimize)
			elif isinstance(coef_dict, ndarray):
				self.model.set_objective(coef_dict, minimize)
			else:
				raise TypeError('`coef_dict` must either be a dict or an ndarray')
		else:
			raise Exception('Cannot set an objective on a model without the optimizer flag as True.')

	def optimize(self, coef_dict=None, minimize=False):
		cur_obj = self.model.model.objective
		if coef_dict != None:
			self.set_objective(coef_dict, minimize)
		sol = self.optimizer.optimize()

		if coef_dict != None:
			self.model.model.objective = cur_obj

		return sol

	def revert_to_original_bounds(self):
		for rx, bounds in zip(self.reaction_names, self.original_bounds):
			clb, cub = self.get_reaction_bounds(rx)
			lb, ub = bounds

			if clb != lb or cub != ub:
				self.set_reaction_bounds(rx, lb=lb, ub=ub)

	def summarize_solution(self, sol, drains, eps=1e-9):
		active_drains = {k:v for k,v in sol.var_values().items() if abs(v) > eps and k in drains}
		table = [[k]+[r for r,v in active_drains.items() if f(v)] for k,f in [('produced',lambda x: x < -eps), ('consumed',lambda x: x > eps)]]
		pretty_table_print(array(table).T.tolist(), has_header=True, header_sep=2)

class CORSOModel(ConstraintBasedModel):
	def __init__(self, cbmodel, corso_element_names=('R_PSEUDO_CORSO', 'M_PSEUDO_CORSO'), solver=None):
		if not cbmodel.model:
			cbmodel.initialize_optimizer()

		self.cbmodel = cbmodel

		irrev_model, self.mapping = cbmodel.make_irreversible()

		S = irrev_model.get_stoichiometric_matrix()
		bounds = irrev_model.bounds

		self.cost_index_mapping = zeros(S.shape[1], dtype=int_)

		self.corso_rx, self.corso_mt = corso_element_names
		super().__init__(S, bounds, irrev_model.reaction_names, irrev_model.metabolite_names, solver=solver)
		self.add_metabolite(zeros(len(self.reaction_names)), self.corso_mt)

		self.add_reaction(zeros(len(self.metabolite_names)), (0, 0), self.corso_rx)

		self.original_bounds = deepcopy(self.bounds)

		for orx, nrx in self.mapping.items():
			if isinstance(nrx, int):
				self.cost_index_mapping[nrx] = orx
			else:
				for nrx_split in nrx:
					self.cost_index_mapping[nrx_split] = orx

	def solve_original_model(self, of_dict, minimize=False):
		self.cbmodel.set_objective(of_dict, minimize)
		sol = self.cbmodel.optimize()
		return sol

	def set_costs(self, cost):
		true_cost = cost[self.cost_index_mapping]
		true_cost = append(true_cost, array([-1]))
		self.set_stoichiometric_matrix(true_cost.reshape(1, len(true_cost)), rows=[self.corso_mt])

	# def set_reaction_bounds(self, index, **kwargs):
	# 	super().set_reaction_bounds(index, **kwargs)
	# 	self.original_bounds[self.decode_index(index, 'reaction')] = self.get_reaction_bounds(index)

	def set_corso_objective(self):
		self.set_objective({self.corso_rx: 1}, True)

	def optimize_corso(self, cost, of_dict, minimize=False, constraint=1, constraintby='val', eps=1e-6, flux1=None):
		if flux1 is None:
			flux1 = self.solve_original_model(of_dict, minimize)

		if abs(flux1.objective_value()) < eps:
			return flux1, flux1
		f1_x = flux1.x()
		if constraintby == 'perc':
			# f1_f = flux1.x()[self.cbmodel.c != 0]
			f1_f = {idx: f1_x[idx] * (constraint / 100) for idx in of_dict.keys()}
		elif constraintby == 'val':
			if (flux1.objective_value() < constraint and not minimize) or (
					flux1.objective_value() > constraint and minimize):
				raise Exception('Objective flux is not sufficient for the the set constraint value.')
			else:
				f1_f = {idx: constraint for idx in of_dict.keys()}
		else:
			raise Exception('Invalid constraint options.')

		self.set_reaction_bounds(self.corso_rx, lb=0, ub=1e20)
		# corso_of_dict = deepcopy(of_dict)
		# corso_of_dict[self.corso_rx] = 1

		self.set_costs(cost)
		for rx in f1_f.keys():
			true_idx = self.decode_index(rx, 'reaction')
			involved = self.mapping[true_idx]
			fluxval = f1_f[rx]  # if isinstance(f1_f, ndarray) else f1_f

			if isinstance(involved, (int, int_)):
				self.set_reaction_bounds(involved, lb=fluxval, ub=fluxval, temporary=True)
			else:
				self.set_reaction_bounds(involved[0], lb=fluxval, ub=fluxval, temporary=True)
				self.set_reaction_bounds(involved[1], lb=0, ub=0, temporary=True)

		self.set_objective({self.corso_rx: 1}, True)

		sol = self.optimize()
		self.revert_to_original_bounds()

		return flux1, CORSOSolution(flux1, sol, sum([of_dict[k] * f1_f[k] for k in of_dict.keys()]),
									self.cost_index_mapping, self.cbmodel.reaction_names, eps=eps)


class GIMMEModel(ConstraintBasedModel):
	def __init__(self, cbmodel, solver=None):
		self.cbmodel = cbmodel
		if not self.cbmodel.model:
			self.cbmodel.initialize_optimizer()

		irrev_model, self.mapping = cbmodel.make_irreversible()

		S = irrev_model.get_stoichiometric_matrix()
		bounds = irrev_model.bounds
		super().__init__(S, bounds, irrev_model.reaction_names, irrev_model.metabolite_names, solver=solver,
						 optimizer=True)

	def __adjust_objective_to_irreversible(self, objective_dict):
		obj_dict = {}
		for k, v in objective_dict.items():
			irrev_map = self.mapping[self.cbmodel.decode_index(k, 'reaction')]
			if isinstance(irrev_map, (list, tuple)):
				for i in irrev_map:
					obj_dict[i] = v
			else:
				obj_dict[irrev_map] = v
		return obj_dict

	def __adjust_expression_vector_to_irreversible(self, exp_vector):
		exp_vector_n = zeros(len(self.reaction_names), )
		for rxn, val in enumerate(exp_vector):
			rxmap = self.mapping[rxn]
			if isinstance(rxmap, tuple):
				exp_vector_n[rxmap[0]] = exp_vector_n[rxmap[1]] = val
			else:
				exp_vector_n[rxmap] = val
		return exp_vector_n

	def optimize_gimme(self, exp_vector, objectives, obj_frac, flux_thres):
		N = len(self.cbmodel.reaction_names)
		objectives_irr = [self.__adjust_objective_to_irreversible(obj) for obj in objectives]
		exp_vector_irr = self.__adjust_expression_vector_to_irreversible(exp_vector)

		def find_objective_value(obj):
			self.cbmodel.set_objective(obj, False)
			return self.cbmodel.optimize().objective_value()

		objective_values = list(map(find_objective_value, objectives_irr))

		gimme_model_objective = array(
			[flux_thres - exp_vector_irr[i] if -1 < exp_vector_irr[i] < flux_thres else 0 for i in range(N)])

		objective_lbs = zeros(len(self.reaction_names))
		for ov, obj in zip(objective_values, objectives_irr):
			for rx, v in obj.items():
				objective_lbs[rx] = v * ov * obj_frac

		objective_ids = nonzero(objective_lbs)[0]
		for id, lb in zip(objective_ids, objective_lbs):
			self.set_reaction_bounds(id, lb=lb, temporary=True)

		self.set_objective(gimme_model_objective, True)
		sol = self.optimize()
		self.revert_to_original_bounds()
		return GIMMESolution(sol, exp_vector, self.cbmodel.reaction_names, self.mapping)
