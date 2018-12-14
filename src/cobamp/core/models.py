from numpy import ndarray, array, where, apply_along_axis
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

			reactions_ok = len(reaction_names) == n and isinstance(reaction_names, dict) or reaction_names is None
			metabolites_ok = len(metabolite_names) == m and isinstance(metabolite_names, dict) or metabolite_names is None

			assert reactions_ok, 'Reaction name dimensions do not match the supplied stoichiometrix matrix'
			assert metabolites_ok, 'Metabolite name dimensions do not match the supplied stoichiometrix matrix'

			return m,n

		m,n = __validate_args()
		#self.__S = self.__parse_stoichiometric_matrix(S)
		self.__S = array(self.S)


		self.bounds = self.__interpret_bounds(thermodynamic_constraints)

		self.reaction_names, self.metabolite_names = None, None
		if reaction_names:
			self.reaction_names = OrderedDict(zip(reaction_names,list(range(n))))
		if metabolite_names:
			self.metabolite_names = OrderedDict(zip(metabolite_names,list(range(m))))

		self.reaction_id_map, self.metabolite_id_map = (OrderedDict([(v,k) for k,v in d]) for d in [self.reaction_names, self.metabolite_names])

		self.inflows, self.outflows = self.__identify_boundaries()

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

	def __decode_index(self, index, label_dict):
		if isinstance(index, int):
			return index
		elif index in label_dict and label_dict is not None:
			return label_dict[index]
		else:
			raise IndexError('Could not find specified index "'+str(index)+'.')

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