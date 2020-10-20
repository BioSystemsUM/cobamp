from cobamp.wrappers.core import AbstractObjectReader
import numpy as np


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
