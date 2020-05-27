from cobamp.utilities.file_io import read_pickle
from cobamp.utilities.tree_analysis import *
from itertools import chain
import matplotlib.pyplot as plt
import networkx as nx

def generate_efm_results_tree(efm_sets, ignore_greater_than=10, pruning_level=6, merge_dupes=False):
 """
 Args:
     efm_sets:
     ignore_greater_than:
     pruning_level:
     merge_dupes:
 """
	root = Tree('ROOT')
	fill_tree(root, efm_sets)
	compress_linear_paths(root)
	if ignore_greater_than:
		ignore_compressed_nodes_by_size(root, ignore_greater_than)
	apply_fx_to_all_node_values(root, lambda x: '\n'.join(sorted(x)) if isinstance(x, list) else x if x is not None else "None")
	if pruning_level:
		probabilistic_tree_prune(root, target_level=pruning_level, cut_leaves=True, name_separator='\n')
	compress_linear_paths(root)
	if merge_dupes:
		merge_duplicate_nodes(root)
	return root

def draw_graph(root, write_path, unique=False):
 """
 Args:
     root:
     write_path:
     unique:
 """
	G = nx.DiGraph()
	populate_nx_graph(root, G, unique_nodes=unique)
	print('NetworkX recognizes this as a tree?',nx.is_tree(G))
	pos = nx.nx_pydot.graphviz_layout(G)
	plt.figure(figsize=(50,40))
	nx.draw_networkx_nodes(G, pos, node_size=0)
	nx.draw_networkx_edges(G, pos, alpha=0.5, arrowsize=10)
	nx.draw_networkx_labels(G, pos, font_size=18, font_color='red')
	if isinstance(write_path, str):
		plt.savefig(write_path)

if __name__ == '__main__':
	efms = read_pickle(
		'/home/skapur/MEOCloud/Projectos/MCSEnumeratorPython/examples/GBconsensus_resources/EFMs/efms_glc_uptake_decoded.pkl')
	efms_orig = [dict(list(chain(*[[(k_token[2:], v) for k_token in k.split('_and_')] for k, v in efm.items()]))) for
				 efm in efms]
	efm_sets = [set([r for r in efm.keys()]) for efm in efms_orig]

	tree = generate_efm_results_tree(
		efm_sets=efm_sets,
		ignore_greater_than=10,
		pruning_level=10,
		merge_dupes=False
	)

	draw_graph(tree, 'test_graph.pdf', unique=False)
