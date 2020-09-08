import numpy as np
import pickle
import pandas as pd
import math

from itertools import product
from cobamp.efm_enumeration.kshortest_efms import KShortestEFMAlgorithm
from cobamp.linear_systems.linear_systems import DualLinearSystem, IrreversibleLinearSystem, SimpleLinearSystem
from cobamp.linear_systems.optimization import LinearSystemOptimizer
from cobamp.mcs_enumeration.intervention_problem import *
from cobamp.utilities.file_io import pickle_object
import cobamp.efm_enumeration.kshortest_efm_properties as kp

#os.chdir('/home/skapur/Workspaces/PyCharm/cobamp')

def decode_mcs(solutions):
 """
 Args:
     solutions:
 """
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

def find_net_conversion(S, efm, met_names, tol=1e-9):
 """
 Args:
     S:
     efm:
     met_names:
     tol:
 """
	digits = abs(round(math.log(tol, 10)))
	metab_balance_dict = {}
	for rx, v in efm.items():
		metabs = np.nonzero(S[:, rx])[0]
		for metab in metabs:
			turnover = v * S[metab, rx]
			if metab not in metab_balance_dict.keys():
				metab_balance_dict[metab] = turnover
			else:
				metab_balance_dict[metab] += turnover

	final_balance_dict = {met_names[k]: round(v, digits) for k, v in metab_balance_dict.items() if abs(v) > tol}
	return final_balance_dict

def enumerate_efms():
	meta_id_from_drain = lambda x: np.nonzero(S[:,list(x)])[0]

	configuration = kp.KShortestProperties()
	configuration[kp.K_SHORTEST_MPROPERTY_METHOD] = kp.K_SHORTEST_METHOD_POPULATE
	configuration[kp.K_SHORTEST_OPROPERTY_MAXSIZE] = 15

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
	irreversible_system = IrreversibleLinearSystem(S_int_final, irrev_int, nc_meta, [consumed_meta[7]], [])
	algorithm = KShortestEFMAlgorithm(configuration)

	efms = algorithm.enumerate(irreversible_system)
	decoded_efms = [{rx_names_int[i]:v for i,v in efm.attribute_value(efm.SIGNED_VALUE_MAP).items() if v != 0} for efm in efms]
	decoded_efms_index = [{i:v for i,v in efm.attribute_value(efm.SIGNED_VALUE_MAP).items() if v != 0} for efm in efms]
	decoded = [' | '.join([rx_names_int[i] for i in efmi.get_active_indicator_varids()]) for efmi in efms]
	net_conversions = [find_net_conversion(S_int_final, efm, met_names_int) for efm in decoded_efms_index]


	efm_set_name = 'efms_glc_uptake_'
	save_folder = 'examples/GBconsensus_resources/EFMs/'

	pickle_object(decoded_efms, save_folder+efm_set_name+"decoded.pkl")
	pickle_object(decoded_efms_index, save_folder+efm_set_name+"decoded_index.pkl")
	pickle_object(net_conversions, save_folder+efm_set_name+"net_conversions.pkl")

	#with open('examples/GBconsensus_resources/EFMs/efms_no_media.pkl','w') as f:
	#	f.write('\n'.join([','.join(d) for d in decoded]))
	#pickle_object(decoded_efms_index, 'examples/GBconsensus_resources/EFMs/efms_no_media_index.pkl')
#	pickle_object(decoded_efms, 'examples/GBconsensus_resources/EFMs/glc_uptake_kmax12.pkl')

	with open('examples/GBconsensus_resources/EFMs/efms_glc_uptake.pkl', 'r') as f:
		decoded_efms = f.read().replace(',',';').split('\n')

	df_dict = {'EFM':[' | '.join(efm) for efm in decoded_efms], 'Conversion':[' | '.join(sorted([str(v)+" "+r for r,v in find_net_conversion(S_int_final, efm, met_names_int).items()])) for efm in decoded_efms_index]}

	df = pd.DataFrame.from_dict(df_dict)
	df.to_csv(save_folder+efm_set_name+"dataframe.csv")
	return decoded


if __name__ == '__main__':
	efm = enumerate_efms()