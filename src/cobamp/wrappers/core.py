import abc
import warnings

import numpy as np
from numpy import where

from cobamp.core.models import ConstraintBasedModel
from cobamp.gpr.core import GPRContainer
from cobamp.utilities.parallel import batch_run, cpu_count

MAX_PRECISION = 1e-10


## TODO: Add simulation and result functions on the wrappers module to automatically detect which one to use
class ConstraintBasedModelSimulator(object):
    def __init__(self, model, simulation_function, result_function):
        self.__model = model
        self.__simulation_function = simulation_function
        self.__result_function = result_function

    def simulate(self, func, bound_change=None, objective_coefficient=None, minimize=None, func_args=None):
        self.__simulation_function(self.__model, func, bound_change, objective_coefficient, minimize,
                                   self.__result_function, func_args)

    def batch_simulate(self, func, bound_changes, objective_coefficients, minimize, func_args=None, mp_threads=None):
        mp_params = {'model':self.__model, 'func_args':func_args}

        def check_multiple_inputs(bound_changes, objective_coefficients, minimize):
            arg_list = bound_changes, objective_coefficients, minimize
            corrected_args = []
            max_len = max(map(len,arg_list))
            for arg in arg_list:
                if len(arg) == 1:
                    corrected_args.append([arg[0]]*max_len)
                elif len(arg) == max_len:
                    corrected_args.append(arg)
                else:
                    raise Exception('One of the arguments contains more than 1 and less than '+str(max_len)+' items.')
            return corrected_args


        def batch_simulation_function(sequence, params):
            model_mp = params['model']
            func_args_mp = params['func_args']
            bc, oc, min = sequence
            return self.__simulation_function(model_mp, func, bc, oc, min, self.__result_function, func_args_mp)

        sequence = list(zip(*check_multiple_inputs(bound_changes, objective_coefficients, minimize)))
        return batch_run(batch_simulation_function, sequence, mp_params,
                         threads=cpu_count() if mp_threads is None else mp_threads)

class AbstractObjectReader(object):
    """
    An abstract class for reading metabolic model objects from external frameworks, and extracting the data needed for
    pathway analysis methods. Also deals with name conversions.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, model, gpr_and_char='and', gpr_or_char='or', gpr_gene_parse_function=str, ttg_ratio=20):
        """
        Parameters

        ----------

            model: A Model instance from the external framework to use. Must be registered in the dict stored as
            external_wrappers.model_readers along with its reader.

        """
        self.model = model
        self.initialize(gpr_and_char, gpr_or_char, gpr_gene_parse_function, ttg_ratio)

    def initialize(self, gpr_and_char='and', gpr_or_char='or', gpr_gene_parse_function=str, ttg_ratio=20):
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
        self.gene_protein_reaction_rules = gpr_and_char, gpr_or_char, gpr_gene_parse_function, ttg_ratio
        self.__gpr_read_params = gpr_and_char, gpr_or_char, gpr_gene_parse_function, ttg_ratio
    @property
    def gene_protein_reaction_rules(self):
        return self.__gene_protein_reaction_rules


    @gene_protein_reaction_rules.setter
    def gene_protein_reaction_rules(self, value):
        and_char, or_char, apply_fx, ttg_ratio = value
        self.__gene_protein_reaction_rules = GPRContainer(
            gpr_list=self.get_model_gpr_strings(),
            and_char=and_char, or_char=or_char, apply_fx=apply_fx, ttg_ratio=ttg_ratio
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
        and_char, or_char, gpr_gene_parse_function, ttg_ratio = self.__gpr_read_params
        ngprs = GPRContainer(
            gpr_list=self.get_model_gpr_strings(),
            and_char=and_char, or_char=or_char, apply_fx=gpr_gene_parse_function, ttg_ratio=ttg_ratio)

        return ConstraintBasedModel(
            S=self.get_stoichiometric_matrix(),
            thermodynamic_constraints=[tuple(float(k) for k in l) for l in self.get_model_bounds()],
            reaction_names=self.r_ids,
            metabolite_names=self.m_ids,
            optimizer= (solver == True) or (solver is not None and solver != False),
            solver=solver if solver not in (True, False) else None,
            gprs=ngprs
        )

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