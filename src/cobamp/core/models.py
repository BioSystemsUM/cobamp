from numpy import ndarray, array, where, apply_along_axis, zeros, vstack, hstack, nonzero, append, int_, int8, int16, \
	int32, int64
from .linear_systems import SteadyStateLinearSystem, VAR_CONTINUOUS
from .optimization import LinearSystemOptimizer, CORSOSolution, GIMMESolution
from collections import OrderedDict
import warnings
from copy import deepcopy

INT_TYPES = (int, int_, int8, int16, int32, int64)
LARGE_NUMBER = 10e6 - 1
SMALL_NUMBER = 1e-6
BACKPREFIX = 'flux_backwards'


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


def make_irreversible_model(S, lb, ub):
	irrev = array([i for i in range(S.shape[1]) if not (lb[i] < 0 and ub[i] > 0)])
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
			nub[new_rx[0]] = abs(lb[orig_rx])
			nub[new_rx[1]] = ub[orig_rx]
		else:
			nlb[new_rx], nub[new_rx] = lb[orig_rx], ub[orig_rx]

	return S_new, nlb, nub, rx_mapping


def _invert_bounds(S, lb, ub):
	irrev_reverse_idx = where((lb <= 0) & (ub <= 0))[0]
	# S[:, irrev_reverse_idx] = -S[:, irrev_reverse_idx]
	temp = ub[irrev_reverse_idx]
	ub[irrev_reverse_idx] = -lb[irrev_reverse_idx]
	lb[irrev_reverse_idx] = 0

	return S, lb, ub


class ConstraintBasedModel(object):
	def __init__(self, S, thermodynamic_constraints, reaction_names=None, metabolite_names=None, optimizer=True,
				 solver=None):

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

		self.c = None
		self.model = None

		if optimizer:
			self.initialize_optimizer()

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
		self.set_objective({reaction:1}, True)
		min_flux = self.optimize().objective_value()
		self.set_objective({reaction:1}, False)
		max_flux = self.optimize().objective_value()

		return min_flux, max_flux

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

	def set_stoichiometric_matrix(self, values, rows=None, columns=None):
		row_index = [self.decode_index(i, 'metabolite') for i in rows] if rows else None
		col_index = [self.decode_index(i, 'reaction') for i in columns] if columns else None

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

			self.model.populate_constraints_from_matrix(values, constraints=constraints, vars=vars)

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
		self.__S = vstack([self.__S, row])
		self.metabolite_names.append(name)

		self.__update_decoder_map()

		if self.model:
			self.model.add_rows_to_model(row.reshape([1, self.__S.shape[1]]), b_lb=array([0]), b_ub=array([0]))

	def add_reaction(self, arg, bounds, name=None):
		assert not name in self.reaction_names, 'Duplicate reaction name found!'
		if isinstance(arg, dict):
			col = zeros([1, self.__S.shape[0]])
			for k, v in arg.items():
				col[0,self.decode_index(k, 'metabolite')] = v
		elif isinstance(arg, ndarray):
			if len(arg) == len(self.metabolite_names):
				col = arg.reshape(self.__S.shape[0], 1)
			else:
				raise Exception('Numpy argument dimensions should be (', len(self.reaction_names), ',). Got', arg.shape,
								'instead')
		else:
			raise ValueError('Invalid argument type: ', type(arg), '. Please supply an ndarray or dict instead.')
		self.__S = hstack([self.__S, col])

		self.reaction_names.append(name)
		self.bounds.append(bounds)

		self.__update_decoder_map()

		if self.model:
			self.model.add_columns_to_model(col, [name], [bounds[0]], [bounds[1]], VAR_CONTINUOUS)

	def remove_reaction(self, index):
		j = self.decode_index(index, 'reaction')
		if not isinstance(index, int):
			self.reaction_names.pop(j)
		self.bounds.pop(j)

		self.__update_decoder_map()

		if self.model:
			self.model.remove_from_model(j, 'var')

	def remove_metabolite(self, index):
		i = self.decode_index(index, 'metabolite')
		if not isinstance(index, int):
			self.metabolite_names.pop(i)

		self.__update_decoder_map()

		if self.model:
			self.model.remove_from_model(i, 'const')

	def make_irreversible(self):
		lb, ub = self.get_bounds_as_list()
		Sn, nlb, nub, rx_mapping = make_irreversible_model(self.__S, lb, ub)
		rev = array([int(v[0]) for k, v in rx_mapping.items() if isinstance(v, (list, tuple))])
		revnames = ['_'.join([name, BACKPREFIX]) for name in (array(self.reaction_names)[rev]).tolist()]
		rnames = self.reaction_names + revnames
		model = ConstraintBasedModel(Sn, list(zip(nlb, nub)), rnames, self.metabolite_names,
									 optimizer=self.model is not None, solver=self.solver)
		return model, rx_mapping

	def get_reaction_bounds(self, index):
		return self.bounds[self.decode_index(index, 'reaction')]

	def set_reaction_bounds(self, index, **kwargs):
		true_idx = self.decode_index(index, 'reaction')
		lb, ub = self.get_reaction_bounds(true_idx)
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
