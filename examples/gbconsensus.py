import numpy as np
import os
import pickle

from itertools import product
from metaconvexpy.convex_analysis.efm_enumeration.kshortest_efms import KShortestEFMAlgorithm
from metaconvexpy.linear_systems.linear_systems import DualLinearSystem, IrreversibleLinearSystem, SimpleLinearSystem
from metaconvexpy.linear_systems.optimization import LinearSystemOptimizer
from metaconvexpy.convex_analysis.mcs_enumeration.intervention_problem import *
from metaconvexpy.utilities.file_utils import pickle_object, read_pickle
import metaconvexpy.convex_analysis.efm_enumeration.kshortest_efm_properties as kp

os.chdir('/home/skapur/Workspaces/PyCharm/metaconvexpy')

def decode_mcs(solutions):
	return list(chain(
		*[list(product(*[orx_map[rx_names[i]] for i in lethal.get_active_indicator_varids()])) for lethal in
		  solutions]))


S = np.genfromtxt('examples/GBconsensus_resources/GBconsensus_comp_stoich.csv', delimiter=',')

with open('examples/GBconsensus_resources/GBconsensus_comp_media.pkl', 'rb') as f:
	media = pickle.load(f)

with open('examples/GBconsensus_resources/GBconsensus_comp_exclusions.txt', 'r') as f:
	singles = [s.strip() for s in f.readlines()]

with open('examples/GBconsensus_resources/GBconsensus_comp_rxnames.txt', 'r') as f:
	rx_names = [s.strip() for s in f.readlines()]

with open('examples/GBconsensus_resources/GBconsensus_comp_metnames.txt', 'r') as f:
	met_names = [s.strip() for s in f.readlines()]

with open('examples/GBconsensus_resources/GBconsensus_comp_bound_map.txt', 'r') as f:
	bound_map = {k: [float(n) for n in v.split(',')] for k, v in
	             dict([s.strip().split('=') for s in f.readlines()]).items()}

with open('examples/GBconsensus_resources/GBconsensus_comp_orx_map.txt', 'r') as f:
	orx_map = {k: [n for n in v.split(',')] for k, v in dict([s.strip().split('=') for s in f.readlines()]).items()}

irrev = np.where(np.array([bound_map[r][0] >= 0 for r in rx_names]))[0]
exclusions = [[rx_names.index([k for k, v in orx_map.items() if s in v][0])] for s in singles if
              s in list(chain(*orx_map.values()))]
biomass_index = rx_names.index('R_biomass_reaction')
glc_index = rx_names.index('R_EX_glc_e')


def enumerate_lethals():

	configuration = kp.KShortestProperties()
	configuration[kp.K_SHORTEST_MPROPERTY_METHOD] = kp.K_SHORTEST_METHOD_ITERATE
	configuration[kp.K_SHORTEST_OPROPERTY_MAXSOLUTIONS] = 1

	problem = InterventionProblem(S)
	target_space = [DefaultFluxbound(0.001, None, biomass_index)] + [DefaultFluxbound(v[0], v[1] if v[1] != 0 else None, rx_names.index(k)) for k, v in
														   media.items() if k in rx_names]
	T, b = problem.generate_target_matrix(target_space)
	dual_system = DualLinearSystem(S, irrev, T, b)

	algorithm = KShortestEFMAlgorithm(configuration)

	lethals = list(algorithm.enumerate(dual_system, exclusions))

	return decode_mcs(lethals)

def optimize_model():
	lin_sys = SimpleLinearSystem(S, [bound_map[k][0] if k not in media.keys() else media[k][0] for k in rx_names], [bound_map[k][1] if k not in media.keys() else media[k][1] for k in rx_names], [n[:128] for n in rx_names])
	lso = LinearSystemOptimizer(lin_sys)
	sol = lso.optimize([('R_biomass_reaction',1)], False)
	sol.var_values()['R_biomass_reaction']

def enumerate_efms():
	meta_id_from_drain = lambda x: np.nonzero(S[:,list(x)])[0]

	configuration = kp.KShortestProperties()
	configuration[kp.K_SHORTEST_MPROPERTY_METHOD] = kp.K_SHORTEST_METHOD_POPULATE
	configuration[kp.K_SHORTEST_OPROPERTY_MAXSIZE] = 10

	drains = set([s for s in singles if "R_EX" in s])

	consumed = set([rx_names.index(k) for k in set(media.keys()) & set(rx_names) if media[k][1] <= 0])
	produced = set([rx_names.index(k) for k in set(media.keys()) & set(rx_names) if media[k][1] > 0])
	non_consumed = set([rx_names.index(k) for k in drains if k in rx_names]) - consumed - produced

	consumed_meta = list(zip(meta_id_from_drain(consumed), [-media[rx_names[i]][0] for i in consumed]))
	produced_meta = meta_id_from_drain(produced)
	nc_meta = meta_id_from_drain(non_consumed)

	drains_id = sorted([rx_names.index(k) for k in drains if k in rx_names])
	not_drains_id = [i for i in range(S.shape[1]) if i not in drains_id]

	S_int = S[:, not_drains_id]
	rx_names_int = [rx_names[i] for i in not_drains_id]
	irrev_int = np.where(np.array([bound_map[r][0] >= 0 for r in rx_names_int]))[0]

	biomass_index = rx_names_int.index('R_biomass_reaction')

	S_int_biomass_meta = np.zeros([1,S_int.shape[1]])

	S_int_biomass_meta[0, biomass_index] = 1

	S_int_final = np.vstack([S_int, S_int_biomass_meta])
	met_names_int = met_names + ['biomass_c']

	produced_meta = [(met_names_int.index('biomass_c'), 1)]
	#consumed_meta = [met_names_int.index('M_glc__D_e')]
	irreversible_system = IrreversibleLinearSystem(S_int_final, irrev_int, nc_meta, [], produced_meta)
	algorithm = KShortestEFMAlgorithm(configuration)

	efms = algorithm.enumerate(irreversible_system)
	decoded_efms = [{rx_names_int[i]:v for i,v in efm.attribute_value(efm.SIGNED_INDICATOR_SUM).items() if v != 0} for efm in efms]
	decoded = [[rx_names_int[i] for i in efmi.get_active_indicator_varids()] for efmi in efms]
#	pickle_object(decoded_efms, 'examples/GBconsensus_resources/EFMs/glc_uptake_kmax12.pkl')

	return decoded


if __name__ == '__main__':
	efm = enumerate_efms()