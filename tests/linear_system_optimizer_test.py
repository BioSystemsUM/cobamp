from cobamp.core.optimization import LinearSystemOptimizer
from cobamp.core.linear_systems import SteadyStateLinearSystem
from cobamp.wrappers.external_wrappers import COBRAModelObjectReader

from cobra.io import read_sbml_model

from urllib.request import urlretrieve
import pandas as pd
import numpy as np

if __name__ == '__main__':

	path, content = urlretrieve('http://bigg.ucsd.edu/static/models/iAF1260.xml')

	model = read_sbml_model(path)
	cobamp_model = COBRAModelObjectReader(model)

	S = cobamp_model.get_stoichiometric_matrix()
	lb, ub = cobamp_model.get_model_bounds(False, True)
	rx_names = cobamp_model.get_reaction_and_metabolite_ids()[0]

	lsystem = SteadyStateLinearSystem(S, lb, ub, rx_names)

	optimizer = LinearSystemOptimizer(lsystem)

	objective_id = rx_names.index('BIOMASS_Ec_iAF1260_core_59p81M')
	f = np.zeros(S.shape[1])
	f[objective_id] = 1

	lsystem.set_objective(f, False)

	cobamp_sol = optimizer.optimize()
	cobra_sol = model.optimize()

	cobra_fluxes = cobra_sol.fluxes
	cobamp_fluxes = pd.Series(cobamp_sol.var_values(),name='new_fluxes')

	sol_df = pd.DataFrame(cobra_fluxes).join(pd.DataFrame(cobamp_fluxes))
	sol_df['diff'] = (sol_df['fluxes'] - sol_df['new_fluxes']) > 1e-10
