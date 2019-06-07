from itertools import combinations, chain
from collections import Counter
from cobamp.analysis.plotting import display_heatmap
import matplotlib.pyplot as plt
import pandas as pd


def get_frequency_dataframe(pathway_dict, k_min=1, k_max=1):
	def _get_possible_combinations(pathway):
		return list(
			chain(*[[' '.join(list(frozenset(c))) for c in combinations(pathway, k)] for k in range(k_min, k_max + 1)]))

	def _get_reaction_frequencies(pathways):
		c = Counter()
		for pathway in pathways:
			c.update(_get_possible_combinations(pathway))
		return c

	return pd.DataFrame(
		{ident: _get_reaction_frequencies(pathways) for ident, pathways in pathway_dict.items()})


if __name__ == '__main__':
	n_reactions = 30
	efm_size_range = (1, 20)
	efm_number = 20
	efm_group_number = 5


	def generate_random_efms(n_reactions, efm_size_range, efm_number, efm_group_number):
		from random import randint
		def random_slightly_readable_string_generator(length):
			s = ""
			vwls = ['a', 'e', 'i', 'o', 'u']
			vwl_flag = bool(randint(0, 1))
			while len(s) < length:
				some_char = chr(randint(97, 122))
				if (some_char in vwls and vwl_flag) or (some_char not in vwls and not vwl_flag):
					s += some_char
					vwl_flag = not vwl_flag
			return s

		reaction_names = [random_slightly_readable_string_generator(randint(4, 10)) for _ in range(n_reactions)]
		group_names = [random_slightly_readable_string_generator(randint(10, 15)) for _ in range(n_reactions)]

		efm_groups = {
			group_names[j]: [set([reaction_names[randint(0, n_reactions - 1)] for i in range(randint(*efm_size_range))])
							 for
							 _ in range(efm_number)] for j in range(efm_group_number)}

		return reaction_names, group_names, efm_groups


	reaction_names, group_names, efm_groups = generate_random_efms(n_reactions, efm_size_range, efm_number,
																   efm_group_number)

	df = get_frequency_dataframe(efm_groups)
	display_heatmap(df)
