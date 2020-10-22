import warnings
import numpy as np

from cobamp.wrappers.core import AbstractObjectReader
from cobra.exceptions import Infeasible, Unbounded, UndefinedSolution

def cobra_simulate(cobra_model, cobra_func, result_func, bound_change, objective_coefficient, minimize, func_args=None):
    if func_args is None:
        func_args = {}
    with cobra_model as context_model:
        if bound_change is not None:
            for k, v in bound_change.items(): context_model.reactions.get_by_id(k).bounds = v

        if objective_coefficient is not None:
            context_model.objective = {context_model.reactions.get_by_id(k):v for k,v in objective_coefficient.items()}

        if minimize is not None:
            context_model.objective.direction = 'min' if minimize else 'max'
        try:
            sol = cobra_func(model=context_model, **func_args)
        except Exception as e:
            if isinstance(e, (Infeasible, Unbounded, UndefinedSolution)):
                sol = None
            else:
                raise e
        finally:
            return result_func(sol)

def cobra_simulation_result_function(sol):
    if sol is not None:
        return (sol.status == 'optimal', sol.objective_value, sol.fluxes.to_dict())
    else:
        return (False, None, None)

def cobra_fba(model, **func_args):
    return model.optimize()

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
        super().__init__(model, gpr_gene_parse_function=gpr_gene_parse_function, **kwargs)

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
