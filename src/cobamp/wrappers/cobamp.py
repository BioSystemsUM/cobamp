from cobamp.wrappers.core import AbstractObjectReader
import numpy as np

def cobamp_simulate(cobamp_model, cobamp_func, bound_change, objective_coefficient, minimize, result_func=None,
                    func_args=None):

    if func_args is None:
        func_args = {}
    if result_func is None:
        result_func = lambda x: x

    with cobamp_model as context_model:
        if bound_change is not None:
            for k, v in bound_change.items(): context_model.set_reaction_bounds(k, lb=v[0], ub=v[1])

        if None not in [objective_coefficient,minimize]:
            context_model.set_objective(objective_coefficient, minimize)

        sol = cobamp_func(model=context_model, **func_args)
        return result_func(sol)

def cobamp_simulation_result_function(sol):
    return sol.status() == 'optimal', sol.objective_value(), sol.to_series().to_dict()

def cobamp_fba(model, **func_args):
    return model.optimize()

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
        return [self.model.gpr[i] for i in range(len(self.r_ids))]