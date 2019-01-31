from numpy import ndarray, array, where, apply_along_axis, zeros, vstack, hstack
from .linear_systems import SteadyStateLinearSystem, VAR_CONTINUOUS
from collections import OrderedDict
import warnings

LARGE_NUMBER = 10e6 - 1
SMALL_NUMBER = 1e-6


class ConstraintBasedModel(object):
	def __init__(self, S, thermodynamic_constraints, reaction_names=None, metabolite_names=None):

		def __validate_args():

			# stoichiometric matrix
			if isinstance(S, ndarray):
				m,n = S.shape
			elif isinstance(S, tuple) or isinstance(S, list):
				m,n = len(S), len(S[0])
			else:
				raise TypeError('S matrix is not an ndarray or list')

			thermodynamic_types = set(map(type,len(thermodynamic_constraints)))
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

		self.reaction_names, self.metabolite_names = reaction_names, metabolite_names

		self.reaction_id_map, self.metabolite_id_map = (OrderedDict([(v,k) for k,v in d]) for d in [self.reaction_names, self.metabolite_names])

		self.inflows, self.outflows = self.__identify_boundaries()

		lb, ub = self.get_bounds_as_list()

		self.model = SteadyStateLinearSystem(S, lb, ub, var_names=self.reaction_names)
		self.model.build_problem()

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
			if value[0] <= value[1]:
				warnings.warn('Lower bound is >= than upper bound for reaction '+str(i)+'. Reversing...')
				value = value[::-1]
			return value

		return list(map(interpret_bound, enumerate(thermodynamic_constraints)))

	# TODO: Maybe find a cleaner way to import the stoich matrix
	#def __parse_stoichiometric_matrix(self, S):

	def __decode_index(self, index, labels):
		if isinstance(index, int):
			return index
		elif index in labels and labels is not None:
			return labels.index(index)
		else:
			raise IndexError('Could not find specified index "'+str(index)+'.')

	def get_bounds_as_list(self):
		return list(zip(self.bounds))
	def is_reversible_reaction(self, index):
		lb, ub = self.bounds[self.__decode_index(index, self.reaction_names)]
		return (lb < 0 and ub > 0)

	def __identify_boundaries(self):
		outflows = where(apply_along_axis(lambda x: sum(x > SMALL_NUMBER), 0, self.__S) == 1)[0]
		inflows = where(apply_along_axis(lambda x: sum(x < -SMALL_NUMBER), 0, self.__S) == 1)[0]
		both = outflows & inflows
		return tuple(outflows - both), tuple(inflows - both)

	def get_reactions_from_metabolite(self, index):
		dec_index = self.__decode_index(index, self.metabolite_names)
		ids = where(self.__S[dec_index,:])[0]
		return tuple(map(lambda x: (self.reaction_id_map[x],self.__S[dec_index,x]), ids))

	def get_metabolites_from_reaction(self, index):
		dec_index = self.__decode_index(index, self.reaction_names)
		ids = where(self.__S[:,dec_index])[0]
		return tuple(map(lambda x: (self.metabolite_id_map[x],self.__S[x,dec_index]), ids))

	def get_stoichiometric_matrix(self, rows=None, columns=None):
		row_index = [self.__decode_index(i, self.metabolite_names) for i in rows] if rows else None
		col_index = [self.__decode_index(i, self.reaction_names) for i in columns] if columns else None

		if rows and columns:
			return self.__S[row_index, col_index]
		elif rows:
			return self.__S[row_index,:]
		elif columns:
			return self.__S[:, col_index]
		else:
			return self.__S

	def add_metabolite(self, arg, name=None):
		assert not name in self.metabolite_names, 'Duplicate metabolite name found!'
		if isinstance(arg, dict):
			row = zeros(1,self.__S.shape[1])
			for k,v in arg.items():
				row[self.__decode_index(k, self.reaction_names)] = v
		elif isinstance(arg, ndarray):
			if len(arg) == len(self.reaction_names):
				row = arg
			else:
				raise Exception('Numpy argument dimensions should be (',len(self.metabolite_names),',). Got',arg.shape,'instead')
		else:
			raise ValueError('Invalid argument type: ',type(arg),'. Please supply an ndarray or dict instead.')
		self.__S = vstack([self.__S, row])
		self.metabolite_names.append(name)

		self.model.add_rows_to_model(row, b_lb=array([0]), b_ub=array([0]))

	def add_reaction(self, arg, bounds, name=None):
		assert not name in self.reaction_names, 'Duplicate reaction name found!'
		if isinstance(arg, dict):
			col = zeros(1,self.__S.shape[0])
			for k,v in arg.items():
				col[self.__decode_index(k, self.metabolite_names)] = v
		elif isinstance(arg, ndarray):
			if len(arg) == len(self.reaction_names):
				col = arg
			else:
				raise Exception('Numpy argument dimensions should be (',len(self.reaction_names),',). Got',arg.shape,'instead')
		else:
			raise ValueError('Invalid argument type: ',type(arg),'. Please supply an ndarray or dict instead.')
		self.__S = hstack([self.__S, col])
		self.reaction_names.append(name)
		self.model.add_columns_to_model(col, [name], [bounds[0]], [bounds[1]], VAR_CONTINUOUS)

	def remove_reaction(self, index):
		j = self.__decode_index(index, self.reaction_names)
		if not isinstance(index, int):
			self.reaction_names.pop(j)
		self.bounds.pop(j)
		self.model.remove_from_model(j, 'var')

	def remove_metabolite(self, index):
		i = self.__decode_index(index, self.metabolite_names)
		if not isinstance(index, int):
			self.metabolite_names.pop(i)
		self.model.remove_from_model(i, 'const')

	def make_irreversible(self):
		lb, ub = self.get_bounds_as_list()
		irrev = array([i for i in range(self.S.shape[1]) if not (lb[i] < 0 and ub[i] > 0)])
		rev = array([i for i in range(self.S.shape[1]) if i not in irrev])
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

		model = ConstraintBasedModel(S_new, list(zip(nlb,nub)))
		return model, rx_mapping

	def set_objective(self, coef_dict, minimize=False):
		f = zeros(len(self.reaction_names))
		for k,v in coef_dict.items():
			f[self.__decode_index(k, self.reaction_names)] = v
		self.model.set_objective(f, minimize)