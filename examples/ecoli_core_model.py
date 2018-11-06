import urllib
import cobra
import escher
from src.cobamp.utilities.external_wrappers import KShortestEFMEnumeratorWrapper

model_url = "http://bigg.ucsd.edu/static/models/e_coli_core.xml"
model_path, model_content = urllib.request.urlretrieve(model_url)
model = cobra.io.sbml3.read_sbml_model(model_path)

def display_efms_escher(efm):
	escher_builder = escher.Builder(
		map_name='e_coli_core.Core metabolism',
		hide_secondary_metabolites = True,
		reaction_data = efm
	)
	escher_builder.display_in_notebook(js_source='local')


if __name__ == '__main__':
	ksefm = KShortestEFMEnumeratorWrapper(
		model=model,
		non_consumed=[],
		consumed=['glc__D_e'],
		produced=['succ_e'],
		algorithm_type=KShortestEFMEnumeratorWrapper.ALGORITHM_TYPE_POPULATE,
		stop_criteria=100
	)

	enumerator = ksefm.get_enumerator()

	efm_list = []
	while len(efm_list) == 0:
		efm_list += next(enumerator)

	display_efms_escher(efm[0])