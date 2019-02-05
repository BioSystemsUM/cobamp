from numpy import ndarray, array, where, apply_along_axis, zeros, vstack, hstack, nonzero, append, int_
from .linear_systems import SteadyStateLinearSystem, VAR_CONTINUOUS
from .optimization import LinearSystemOptimizer, CORSOSolution
from collections import OrderedDict
import warnings
from copy import deepcopy

LARGE_NUMBER = 10e6 - 1
SMALL_NUMBER = 1e-6
BACKPREFIX = 'flux_backwards'


def make_irreversible_model(S, lb, ub):
	irrev = array([i for i in range(S.shape[1]) if not (lb[i] < 0 and ub[i] > 0)])
	rev = array([i for i in range(S.shape[1]) if i not in irrev])
	Sr = S[:, rev]
	offset = S.shape[1]
	rx_mapping = {k: v if k in irrev else [v] for k, v in dict(zip(range(offset), range(offset))).items()}
	for i, rx in enumerate(rev):
		rx_mapping[rx].append(offset + i)
	rx_mapping = {k: tuple(v) if isinstance(v, list) else v for k, v in rx_mapping.items()}

	S_new = hstack([S, -Sr])
	nlb, nub = zeros(S_new.shape[1]), zeros(S_new.shape[1])
	for orig_rx, new_rx in rx_mapping.items():
		if isinstance(new_rx, tuple):
			nub[new_rx[0]] = abs(lb[orig_rx])
			nub[new_rx[1]] = ub[orig_rx]
		else:
			nlb[new_rx], nub[new_rx] = lb[orig_rx], ub[orig_rx]

	return S_new, nlb, nub, rx_mapping


class ConstraintBasedModel(object):
	def __init__(self, S, thermodynamic_constraints, reaction_names=None, metabolite_names=None, optimizer=True):

		def __validate_args():

			# stoichiometric matrix
			if isinstance(S, ndarray):
				m,n = S.shape
			elif isinstance(S, tuple) or isinstance(S, list):
				m,n = len(S), len(S[0])
			else:
				raise TypeError('S matrix is not an ndarray or list')

			thermodynamic_types = set(map(type,thermodynamic_constraints))
			allowed_types = {list, tuple, bool}

			assert len(thermodynamic_types & allowed_types) == len(thermodynamic_types), 'Invalid bound type found.'

			reactions_ok = len(reaction_names) == n and isinstance(reaction_names, list) or reaction_names is None
			metabolites_ok = len(metabolite_names) == m and isinstance(metabolite_names, list) or metabolite_names is None

			assert reactions_ok, 'Reaction name dimensions do not match the supplied stoichiometrix matrix'
			assert metabolites_ok, 'Metabolite name dimensions do not match the supplied stoichiometrix matrix'

			return m,n

		m,n = __validate_args()
		#self.__S = self.__parse_stoichiometric_matrix(S)
		self.__S = array(S)


		self.bounds = self.__interpret_bounds(thermodynamic_constraints)
		self.original_bounds = deepcopy(self.bounds)
		self.reaction_names, self.metabolite_names = reaction_names, metabolite_names

#		self.reaction_id_map, self.metabolite_id_map = (OrderedDict([(v,k) for k,v in d]) for d in [self.reaction_names, self.metabolite_names])

		self.inflows, self.outflows = self.__identify_boundaries()

		self.c = None
		self.model = None

		if optimizer:
			self.initialize_optimizer()

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
				warnings.warn('Lower bound is >= than upper bound for reaction '+str(i)+'. Reversing...')
				value = value[::-1]
			return value

		return list(map(interpret_bound, enumerate(thermodynamic_constraints)))

	# TODO: Maybe find a cleaner way to import the stoich matrix
	#def __parse_stoichiometric_matrix(self, S):
	def initialize_optimizer(self):
		lb, ub = self.get_bounds_as_list()
		self.model = SteadyStateLinearSystem(self.__S, lb, ub, var_names=self.reaction_names)
		self.model.build_problem()
		self.optimizer = LinearSystemOptimizer(self.model, build=False)

	def decode_index(self, index, labels):
		if type(index) in [int_, int]:
			return index
		elif index in labels and labels is not None:
			return labels.index(index)
		else:
			raise IndexError('Could not find specified index "'+str(index)+'".')

	def get_bounds_as_list(self):
		return list(zip(*self.bounds))

	def is_reversible_reaction(self, index):
		lb, ub = self.bounds[self.decode_index(index, self.reaction_names)]
		return (lb < 0 and ub > 0)

	def __identify_boundaries(self):
		outflows = set(where(apply_along_axis(lambda x: sum(x > SMALL_NUMBER), 0, self.__S) == 1)[0])
		inflows = set(where(apply_along_axis(lambda x: sum(x < -SMALL_NUMBER), 0, self.__S) == 1)[0])
		both = outflows | inflows
		return tuple(outflows - both), tuple(inflows - both)

	def get_reactions_from_metabolite(self, index):
		dec_index = self.decode_index(index, self.metabolite_names)
		ids = where(self.__S[dec_index,:])[0]
		return tuple(map(lambda x: (self.reaction_id_map[x],self.__S[dec_index,x]), ids))

	def get_metabolites_from_reaction(self, index):
		dec_index = self.decode_index(index, self.reaction_names)
		ids = where(self.__S[:,dec_index])[0]
		return tuple(map(lambda x: (self.metabolite_id_map[x],self.__S[x,dec_index]), ids))

	def get_stoichiometric_matrix(self, rows=None, columns=None):
		row_index = [self.decode_index(i, self.metabolite_names) for i in rows] if rows else None
		col_index = [self.decode_index(i, self.reaction_names) for i in columns] if columns else None

		if rows and columns:
			return self.__S[row_index, col_index]
		elif rows:
			return self.__S[row_index,:]
		elif columns:
			return self.__S[:, col_index]
		else:
			return self.__S

	def set_stoichiometric_matrix(self, values, rows=None, columns=None):
		row_index = [self.decode_index(i, self.metabolite_names) for i in rows] if rows else None
		col_index = [self.decode_index(i, self.reaction_names) for i in columns] if columns else None

		if rows and columns:
			self.__S[row_index, col_index] = values
		elif rows:
			self.__S[row_index,:] = values
		elif columns:
			self.__S[:, col_index] = values
		else:
			self.__S = values

		row_index = row_index if row_index is not None else range(self.__S.shape[0])
		col_index = col_index if col_index is not None else range(self.__S.shape[1])

		if self.model:
			constraints = [self.model.model.constraints[i] for i in row_index] if row_index else self.model.model.constraints
			vars = [self.model.model.variables[i] for i in col_index] if row_index else self.model.model.variables

			self.model.populate_constraints_from_matrix(values, constraints=constraints, vars=vars)


	def add_metabolite(self, arg, name=None):
		assert not name in self.metabolite_names, 'Duplicate metabolite name found!'
		if isinstance(arg, dict):
			row = zeros(1,self.__S.shape[1])
			for k,v in arg.items():
				row[self.decode_index(k, self.reaction_names)] = v
		elif isinstance(arg, ndarray):
			if len(arg) == len(self.reaction_names):
				row = arg
			else:
				raise Exception('Numpy argument dimensions should be (',len(self.metabolite_names),',). Got',arg.shape,'instead')
		else:
			raise ValueError('Invalid argument type: ',type(arg),'. Please supply an ndarray or dict instead.')
		self.__S = vstack([self.__S, row])
		self.metabolite_names.append(name)

		if self.model:
			self.model.add_rows_to_model(row.reshape([1, self.__S.shape[1]]), b_lb=array([0]), b_ub=array([0]))

	def add_reaction(self, arg, bounds, name=None):
		assert not name in self.reaction_names, 'Duplicate reaction name found!'
		if isinstance(arg, dict):
			col = zeros(1,self.__S.shape[0])
			for k,v in arg.items():
				col[self.decode_index(k, self.metabolite_names)] = v
		elif isinstance(arg, ndarray):
			if len(arg) == len(self.metabolite_names):
				col = arg.reshape(self.__S.shape[0],1)
			else:
				raise Exception('Numpy argument dimensions should be (',len(self.reaction_names),',). Got',arg.shape,'instead')
		else:
			raise ValueError('Invalid argument type: ',type(arg),'. Please supply an ndarray or dict instead.')
		self.__S = hstack([self.__S, col])
		self.reaction_names.append(name)
		self.bounds.append(bounds)

		if self.model:
			self.model.add_columns_to_model(col, [name], [bounds[0]], [bounds[1]], VAR_CONTINUOUS)

	def remove_reaction(self, index):
		j = self.decode_index(index, self.reaction_names)
		if not isinstance(index, int):
			self.reaction_names.pop(j)
		self.bounds.pop(j)
		if self.model:
			self.model.remove_from_model(j, 'var')

	def remove_metabolite(self, index):
		i = self.decode_index(index, self.metabolite_names)
		if not isinstance(index, int):
			self.metabolite_names.pop(i)

		if self.model:
			self.model.remove_from_model(i, 'const')

	def make_irreversible(self):
		lb, ub = self.get_bounds_as_list()
		irrev = array([i for i in range(self.__S.shape[1]) if not (lb[i] < 0 and ub[i] > 0)])
		rev = array([i for i in range(self.__S.shape[1]) if i not in irrev])
		Sr = self.__S[:, rev]
		offset = self.__S.shape[1]
		rx_mapping = {k: v if k in irrev else [v] for k, v in dict(zip(range(offset), range(offset))).items()}
		for i, rx in enumerate(rev):
			rx_mapping[rx].append(offset + i)
		rx_mapping = {k: tuple(v) if isinstance(v, list) else v for k, v in rx_mapping.items()}

		S_new = hstack([self.__S, -Sr])
		nlb, nub = zeros(S_new.shape[1]), zeros(S_new.shape[1])
		for orig_rx, new_rx in rx_mapping.items():
			if isinstance(new_rx, tuple):
				nub[new_rx[0]] = abs(lb[orig_rx])
				nub[new_rx[1]] = ub[orig_rx]
			else:
				nlb[new_rx], nub[new_rx] = lb[orig_rx], ub[orig_rx]

		revnames = ['_'.join([name,BACKPREFIX]) for name in (array(self.reaction_names)[rev]).tolist()]
		rnames = self.reaction_names + revnames
		model = ConstraintBasedModel(S_new, list(zip(nlb,nub)), rnames, self.metabolite_names, optimizer=self.model is not None)
		return model, rx_mapping

	def get_reaction_bounds(self, index):
		return self.bounds[self.decode_index(index, self.reaction_names)]

	def set_reaction_bounds(self, index, **kwargs):
		true_idx = self.decode_index(index, self.reaction_names)
		lb, ub = self.get_reaction_bounds(true_idx)
		if 'lb' in kwargs:
			lb = kwargs['lb']
		if 'ub' in kwargs:
			ub = kwargs['ub']
		bound = (lb,ub)
		self.bounds[true_idx] = bound

		if self.model:
			var = self.model.model.variables[true_idx]
			self.model.set_variable_bounds([var],[lb],[ub])


	def set_objective(self, coef_dict, minimize=False):
		if self.model:
			f = zeros(len(self.reaction_names))
			self.c = f
			for k,v in coef_dict.items():
				f[self.decode_index(k, self.reaction_names)] = v
			self.model.set_objective(f, minimize)
		else:
			raise Exception('Cannot set an objective on a model without the optimizer flag as True.')

	def optimize(self):
		return self.optimizer.optimize()

class CORSOModel(ConstraintBasedModel):
	def __init__(self, cbmodel, corso_element_names=('R_PSEUDO_CORSO', 'M_PSEUDO_CORSO')):
		if not cbmodel.model:
			cbmodel.initialize_optimizer()

		self.cbmodel = cbmodel

		irrev_model, self.mapping = cbmodel.make_irreversible()

		S = irrev_model.get_stoichiometric_matrix()
		bounds = irrev_model.bounds

		self.cost_index_mapping = zeros(S.shape[1], dtype=int_)

		self.corso_rx, self.corso_mt = corso_element_names
		super().__init__(S, bounds, irrev_model.reaction_names, irrev_model.metabolite_names)
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

	def revert_to_original_bounds(self):
		for rx,bounds in zip(self.reaction_names, self.original_bounds):
			lb, ub = bounds
			self.set_reaction_bounds(rx, lb=lb, ub=ub)

	def set_costs(self, cost):
		#true_cost = zeros(len(self.reaction_names))
		true_cost = cost[self.cost_index_mapping]
		true_cost = append(true_cost,array([-1]))
		self.set_stoichiometric_matrix(true_cost.reshape(1, len(true_cost)), rows=[self.corso_mt])


	def set_corso_objective(self):
		self.set_objective({self.corso_rx:1}, True)

	def optimize_corso(self, cost, of_dict, minimize=False, constraint=1, constraintby='val', eps=1e-6):
		flux1 = self.solve_original_model(of_dict, minimize)

		if abs(flux1.objective_value()) < eps:
			return flux1, flux1

		if constraintby == 'perc':
			f1_f = flux1.x()[self.cbmodel.c != 0]*(constraint/100)
		elif constraintby == 'val':
			if (flux1.objective_value() < constraint and not minimize) or (flux1.objective_value() > constraint and minimize):
				raise Exception('Objective flux is not sufficient for the the set constraint value.')
			else:
				f1_f = constraint
		else:
			raise Exception('Invalid constraint options.')

		self.set_reaction_bounds(self.corso_rx, lb=0, ub=1e20)
		corso_of_dict = deepcopy(of_dict)
		corso_of_dict[self.corso_rx] = 1

		self.set_costs(cost)
		for i,rx in enumerate(nonzero(f1_f)[0]):
			true_idx = self.decode_index(rx, self.reaction_names)
			involved = self.mapping[true_idx]
			fluxval = f1_f[i] if isinstance(f1_f, ndarray) else f1_f

			if isinstance(involved, (int,int_)):
				self.set_reaction_bounds(involved, lb=fluxval, ub=fluxval)
			else:
				self.set_reaction_bounds(involved[0], lb=fluxval, ub=fluxval)
				self.set_reaction_bounds(involved[1], lb=fluxval, ub=fluxval)

		self.set_objective(corso_of_dict, True)

		sol = self.optimize()
		self.revert_to_original_bounds()

		return flux1, CORSOSolution(sol, f1_f, self.cost_index_mapping, self.cbmodel.reaction_names)



