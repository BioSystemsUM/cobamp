import abc
import warnings

import numpy as np
from numpy import where

from ..core.models import ConstraintBasedModel
from ..gpr.evaluator import GPREvaluator

MAX_PRECISION = 1e-10

class AbstractObjectReader(object):
	"""
	An abstract class for reading metabolic model objects from external frameworks, and extracting the data needed for
	pathway analysis methods. Also deals with name conversions.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, model, gpr_and_char='and', gpr_or_char='or', gpr_gene_parse_function=str):
		"""
		Parameters

		----------

			model: A Model instance from the external framework to use. Must be registered in the dict stored as
			external_wrappers.model_readers along with its reader.

		"""
		self.model = model
		self.initialize()

	def initialize(self, gpr_and_char='and', gpr_or_char='or', gpr_gene_parse_function=str):
		"""
			This method re-initializes the class attributes from the current state of self.model
		"""

		self.r_ids, self.m_ids = self.get_reaction_and_metabolite_ids()
		self.rx_instances = self.get_rx_instances()
		self.S = self.get_stoichiometric_matrix()
		self.lb, self.ub = tuple(zip(*self.get_model_bounds(False)))
		self.irrev_bool = self.get_irreversibilities(False)
		self.irrev_index = self.get_irreversibilities(True)
		self.bounds_dict = self.get_model_bounds(True)
		self.gene_protein_reaction_rules = gpr_and_char, gpr_or_char, gpr_gene_parse_function

	@property
	def gene_protein_reaction_rules(self):
		return self.__gene_protein_reaction_rules


	@gene_protein_reaction_rules.setter
	def gene_protein_reaction_rules(self, value):
		and_char, or_char, apply_fx = value
		self.__gene_protein_reaction_rules = GPREvaluator(
			gpr_list=self.get_model_gpr_strings(),
			and_char=and_char, or_char=or_char, apply_fx=apply_fx
		)

	@abc.abstractmethod
	def get_stoichiometric_matrix(self):
		"""
		Returns a 2D numpy array with the stoichiometric matrix whose metabolite and reaction indexes match the names
		defined in the class variables r_ids and m_ids
		"""
		pass

	@abc.abstractmethod
	def get_model_bounds(self, as_dict, separate_list):
		"""
		Returns the lower and upper bounds for all fluxes in the model. This either comes in the form of an ordered list
		with tuples of size 2 (lb,ub) or a dictionary with the same tuples mapped by strings with reaction identifiers.

		Parameters

		----------

			as_dict: A boolean value that controls whether the result is a dictionary mapping str to tuple of size 2
			separate: A boolean value that controls whether the result is two numpy.array(), one for lb and the other\n
			to ub
		"""
		pass

	@abc.abstractmethod
	def get_irreversibilities(self, as_index):
		"""
		Returns a vector representing irreversible reactions, either as a vector of booleans (each value is a flux,
		ordered in the same way as reaction identifiers) or as a vector of reaction indexes.

		Parameters

		----------

			as_dict: A boolean value that controls whether the result is a vector of indexes

		"""
		pass

	@abc.abstractmethod
	def get_rx_instances(self):
		"""
		Returns the reaction instances contained in the model. Varies depending on the framework.
		"""
		pass

	@abc.abstractmethod
	def get_reaction_and_metabolite_ids(self):
		"""
		Returns two ordered iterables containing the metabolite and reaction ids respectively.
		"""
		pass

	# @abc.abstractmethod
	# def get_model_genes(self):
	# 	"""
	#
	# 	Returns the identifiers for the genes contained in the model
	#
	# 	"""

	@property
	def genes(self):
		return self.gene_protein_reaction_rules.get_genes()


	def reaction_id_to_index(self, id):
		"""
		Returns the numerical index of a reaction when given a string representing its identifier.

		Parameters

		----------

			id: A reaction identifier as a string

		"""
		return self.r_ids.index(id)

	def metabolite_id_to_index(self, id):
		"""
		Returns the numerical index of a metabolite when given a string representing its identifier.

		Parameters

		----------

			id: A metabolite identifier as a string

		"""
		return self.m_ids.index(id)

	def get_gene_protein_reaction_rule(self, id):
		return self.gene_protein_reaction_rules[id]

	def convert_constraint_ids(self, tup, yield_constraint):
		if yield_constraint:
			constraint = tuple(list(map(self.reaction_id_to_index, tup[:2])) + list(tup[2:]))
		else:
			constraint = tuple([self.reaction_id_to_index(tup[0])] + list(tup[1:]))
		return constraint


	def get_reaction_scores(self, expression, and_fx=min, or_fx=max, as_vector=False):
		exp_map = {rx: self.gene_protein_reaction_rules.eval_gpr(i, expression, or_fx=or_fx, and_fx=and_fx)
				   for i,rx in enumerate(self.r_ids)}

		if as_vector:
			return [exp_map[k] for k in self.r_ids]
		else:
			return exp_map

	#@warnings.warn('g2rx will be deprecated in a future release. Use the get_reaction_scores method instead',
	#			   DeprecationWarning)
	def g2rx(self, expression, and_fx=min, or_fx=max, as_vector=False, apply_fx=str):
		warnings.warn('g2rx will be deprecated in a future release. Use the get_reaction_scores method instead',
					   DeprecationWarning)
		return self.get_reaction_scores(expression, and_fx, or_fx, as_vector)


	@abc.abstractmethod
	def get_model_gpr_strings(self):
		pass


	def to_cobamp_cbm(self, solver=None):
		return ConstraintBasedModel(
			S=self.get_stoichiometric_matrix(),
			thermodynamic_constraints=[tuple(float(k) for k in l) for l in self.get_model_bounds()],
			reaction_names=self.r_ids,
			metabolite_names=self.m_ids,
			optimizer= solver == True,
			solver=solver if solver not in (True, False) else None,
			gprs=self.gene_protein_reaction_rules
		)

class COBRAModelObjectReader(AbstractObjectReader):

	def __read_model(self, path, format, **kwargs):
		from cobra.io import read_sbml_model, load_matlab_model, load_json_model
		parse_functions = {
			'xml': read_sbml_model,
			'mat': load_matlab_model,
			'json': load_json_model,
			'sbml': read_sbml_model
		}
		if format == None:
			nformat = path.split('.')[-1]
		else:
			nformat = format
		if nformat in parse_functions.keys():
			return parse_functions[nformat](path, **kwargs)
		else:
			raise ValueError('Format '+str(nformat)+' is invalid or not yet available through the cobrapy readers. '+
							 'Choose one of the following: '+','.join(parse_functions.keys()))

	def __init__(self, model, gpr_gene_parse_function=str, format=None, **kwargs):
		if isinstance(model, str):
			warnings.warn('Reading model with cobrapy from the provided path...')
			model = self.__read_model(model, format, **kwargs)
		super().__init__(model, gpr_gene_parse_function=gpr_gene_parse_function)

	def get_stoichiometric_matrix(self):
		S = np.zeros((len(self.m_ids), len(self.r_ids)))
		for i, r_id in enumerate(self.r_ids):
			for metab, coef in self.model.reactions.get_by_id(r_id).metabolites.items():
				S[self.m_ids.index(metab.id), i] = coef

		return S

	def get_model_bounds(self, as_dict=False, separate_list=False):
		bounds = [r.bounds for r in self.rx_instances]
		if as_dict:
			return dict(zip(self.r_ids, bounds))
		else:
			if separate_list:
				return [list(bounds) for bounds in list(zip(*tuple(bounds)))]
			else:
				return tuple(bounds)

	def get_irreversibilities(self, as_index):
		irrev = [not r.reversibility for r in self.rx_instances]
		if as_index:
			irrev = list(np.where(irrev)[0])
		return irrev

	def get_rx_instances(self):
		return [self.model.reactions.get_by_id(rx) for rx in self.r_ids]

	def get_reaction_and_metabolite_ids(self):
		return tuple([[x.id for x in lst] for lst in (self.model.reactions, self.model.metabolites)])

	def get_model_genes(self):
		return set([g.id for g in self.model.genes])

	def get_model_gpr_strings(self, apply_fx=None):
		return [apply_fx(r.gene_reaction_rule) if apply_fx is not None
				else r.gene_reaction_rule for r in self.model.reactions]


class MatFormatReader(AbstractObjectReader):
	def get_stoichiometric_matrix(self):
		return (self.model['S'][0][0]).toarray()

	def get_model_bounds(self, as_dict=False, separate_list=False):
		lb, ub = [(self.model[k][0][0]).ravel() for k in ('lb', 'ub')]
		tuples = [(r, (l, u)) for r, l, u in zip(self.r_ids, lb, ub)]
		if as_dict:
			return dict(tuples)
		else:
			if separate_list:
				return lb, ub
			else:
				return tuple([(l, u) for l, u in zip(lb, ub)])

	def get_irreversibilities(self, as_index):
		if 'rev' in self.model.dtype.names:
			bv = (self.model['rev'][0][0]).ravel().astype(bool)
		else:
			bv = np.array([(l >= 0 and u >= 0) or (l <= 0 and u <= 0) for l, u in zip(self.lb, self.ub)]).astype(bool)
		if as_index:
			return where(bv)[0]
		else:
			return bv

	def get_rx_instances(self):
		pass

	def get_reaction_and_metabolite_ids(self):
		return [[k[0][0] for k in self.model[t][0][0]] for t in ['rxns', 'mets']]

	def get_model_genes(self):
		return set([k[0][0] for k in self.model['genes'][0][0]])

	def get_model_gpr_strings(self):
		return [k[0][0] if len(k[0]) > 0 else '' for k in self.model['grRules'][0][0]]


class FramedModelObjectReader(AbstractObjectReader):

	def get_stoichiometric_matrix(self):
		return np.array(self.model.stoichiometric_matrix())

	def get_model_bounds(self, as_dict=False, separate_list=False):
		bounds = [(r.lb, r.ub) for r in self.rx_instances]
		if as_dict:
			return dict(zip(self.r_ids, bounds))
		else:
			if separate_list:
				return [bounds for bounds in list(zip(*tuple(bounds)))]
			else:
				return tuple(bounds)

	def get_irreversibilities(self, as_index):
		irrev = [not r.reversible for r in self.rx_instances]
		if as_index:
			irrev = list(np.where(irrev)[0])
		return irrev

	def get_reaction_and_metabolite_ids(self):
		return tuple(self.model.reactions.keys()), tuple(self.model.metabolites.keys())

	def get_rx_instances(self):
		return [self.model.reactions[rx] for rx in self.r_ids]

	def get_model_gpr_strings(self):
		return [rx.gene_reaction_rule for rx in self.get_rx_instances()]


class CobampModelObjectReader(AbstractObjectReader):

	def get_stoichiometric_matrix(self):
		return self.model.get_stoichiometric_matrix()

	def get_model_bounds(self, as_dict, separate_list=False):
		if as_dict:
			return dict(zip(self.r_ids, self.model.bounds))
		else:
			if separate_list:
				return [bounds for bounds in list(zip(*tuple(self.model.bounds)))]
			else:
				return tuple(self.model.bounds)

	def get_irreversibilities(self, as_index):
		irrev = [not self.model.is_reversible_reaction(r) for r in self.r_ids]
		if as_index:
			irrev = list(np.where(irrev)[0])
		return irrev

	def get_reaction_and_metabolite_ids(self):
		return self.model.reaction_names, self.model.metabolite_names

	def get_rx_instances(self):
		return None

	def get_model_gpr_strings(self):
		return [self.gene_protein_reaction_rules[i] for i in range(len(self.r_ids))]

# This dict contains the mapping between model instance namespaces (without the class name itself) and the appropriate
# model reader object. Modify this if a new reader is implemented.

model_readers = {
	'cobra.core.model': COBRAModelObjectReader,
	'framed.model.cbmodel': FramedModelObjectReader,
	'cobamp.core.models': CobampModelObjectReader,
	'numpy': MatFormatReader
}
