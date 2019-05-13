from cobamp.algorithms.kshortest import InterventionProblem

import abc
import numpy as np
from scipy.io import loadmat
from numpy import where
from ..core.models import ConstraintBasedModel

MAX_PRECISION = 1e-10


class AbstractObjectReader(object):
	"""
	An abstract class for reading metabolic model objects from external frameworks, and extracting the data needed for
	pathway analysis methods. Also deals with name conversions.
	"""
	__metaclass__ = abc.ABCMeta

	def __init__(self, model):
		"""
		Parameters

		----------

			model: A Model instance from the external framework to use. Must be registered in the dict stored as
			external_wrappers.model_readers along with its reader.

		"""
		self.model = model
		self.r_ids, self.m_ids = self.get_reaction_and_metabolite_ids()
		self.rx_instances = self.get_rx_instances()
		self.S = self.get_stoichiometric_matrix()
		self.lb, self.ub = tuple(zip(*self.get_model_bounds(False)))
		self.irrev_bool = self.get_irreversibilities(False)
		self.irrev_index = self.get_irreversibilities(True)
		self.bounds_dict = self.get_model_bounds(True)
		self.genes = self.get_model_genes()
		self.gene_protein_reaction_rules = self.get_model_gprs()

	@abc.abstractmethod
	def get_stoichiometric_matrix(self):
		"""
		Returns a 2D numpy array with the stoichiometric matrix whose metabolite and reaction indexes match the names
		defined in the class variables r_ids and m_ids
		"""
		return

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
		return

	@abc.abstractmethod
	def get_irreversibilities(self, as_index):
		"""
		Returns a vector representing irreversible reactions, either as a vector of booleans (each value is a flux,
		ordered in the same way as reaction identifiers) or as a vector of reaction indexes.

		Parameters

		----------

			as_dict: A boolean value that controls whether the result is a vector of indexes

		"""
		return

	@abc.abstractmethod
	def get_rx_instances(self):
		"""
		Returns the reaction instances contained in the model. Varies depending on the framework.
		"""
		return

	@abc.abstractmethod
	def get_reaction_and_metabolite_ids(self):
		"""
		Returns two ordered iterables containing the metabolite and reaction ids respectively.
		"""
		return

	@abc.abstractmethod
	def get_model_genes(self):
		"""

		Returns the identifiers for the genes contained in the model

		"""

	@abc.abstractmethod
	def get_model_gprs(self, apply_fx=None):
		"""

		Returns the model gene-protein-reaction rules associated with each reaction

		"""

	@abc.abstractmethod
	def convert_gprs_to_list(self, rx, apply_fx):
		return

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

	def decode_k_shortest_solution(self, solarg):
		if isinstance(solarg, list):
			return [self.__decode_k_shortest_solution(sol) for sol in solarg]
		else:
			return self.__decode_k_shortest_solution(solarg)

	def __decode_k_shortest_solution(self, sol):
		## TODO: Make MAX_PRECISION a parameter for linear systems or the KShortestAlgorithm
		return {self.r_ids[k]: v for k, v in sol.attribute_value(sol.SIGNED_VALUE_MAP).items() if
				abs(v) > MAX_PRECISION}

	def g2rx(self, expression, and_fx=min, or_fx=max, as_vector=False, apply_fx=None):
		gpr_map = {rx: self.convert_gprs_to_list(rx, apply_fx) for rx in self.r_ids}

		def aux_apply(fx, it):
			args = [k for k in it if k is not None]
			return fx(args) if args else None

		def eval_gpr(lists):
			return aux_apply(or_fx,
							 [aux_apply(and_fx, [expression[x] for x in gs if x in expression.keys()]) for gs in lists])

		exp_map = {rx: eval_gpr(gpr_map[rx]) for rx in gpr_map}

		if as_vector:
			return [exp_map[k] for k in self.r_ids]
		else:
			return exp_map

	def to_cobamp_cbm(self, solver=None):
		return ConstraintBasedModel(
			S = self.get_stoichiometric_matrix(),
			thermodynamic_constraints=[tuple(float(k) for k in l) for l in self.get_model_bounds()],
			reaction_names=self.r_ids,
			metabolite_names=self.m_ids,
			optimizer= solver != None and solver,
			solver=solver if solver not in (True, False) else None)

class COBRAModelObjectReader(AbstractObjectReader):

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

	def get_model_gprs(self, apply_fx=None):
		if not apply_fx:
			return [r.gene_reaction_rule for r in self.model.reactions]
		else:
			return [apply_fx(r.gene_reaction_rule) for r in self.model.reactions]

	def convert_gprs_to_list(self, rx, apply_fx):
		proteins = list(map(lambda x: x.strip().replace('(', '').replace(')', ''),
							self.get_model_gprs(apply_fx)[self.r_ids.index(rx)].split('or')))
		rules = [[s.strip() for s in x.split('and') if s.strip() != ''] for x in proteins]
		return rules


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
			bv = np.array([(l >= 0 and u >= 0) or (l <= 0 and u <= 0) for l,u in zip(self.lb, self.ub)]).astype(bool)
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

	def get_model_gprs(self, apply_fx=None):
		gprs = [k[0][0] if len(k[0]) > 0 else '' for k in self.model['grRules'][0][0]]
		if apply_fx:
			return list(map(apply_fx, gprs))
		else:
			return gprs

	def convert_gprs_to_list(self, rx, apply_fx):
		proteins = list(map(lambda x: x.strip().replace('(', '').replace(')', ''),
							self.get_model_gprs(apply_fx)[self.r_ids.index(rx)].split('or')))
		rules = [[s.strip() for s in x.split('and') if s.strip() != ''] for x in proteins]
		return rules


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

	def convert_gprs_to_list(self, rx):
		proteins = list(map(lambda x: x.strip().replace('(', '').replace(')', ''), rx.gene_reaction_rule.split('or')))
		rules = [[s.strip() for s in x.split('and') if s.strip() != ''] for x in proteins]
		return rules


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


# This dict contains the mapping between model instance namespaces (without the class name itself) and the appropriate
# model reader object. Modify this if a new reader is implemented.

model_readers = {
	'cobra.core.model': COBRAModelObjectReader,
	'framed.model.cbmodel': FramedModelObjectReader,
	'cobamp.core.models': CobampModelObjectReader,
	'numpy': MatFormatReader
}
