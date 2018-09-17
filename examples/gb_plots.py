from escher import Builder
from metaconvexpy.utilities.file_utils import read_pickle, pickle_object
from itertools import chain
from collections import Counter
import json


efms = read_pickle('/home/skapur/MEOCloud/Projectos/MCSEnumeratorPython/examples/GBconsensus_resources/EFMs/efms_glc_uptake_decoded.pkl')
#efms_orig = [dict(list(chain(*[[(k_token[2:],v) for k_token in k.split('_and_')] for k,v in efm.items()]))) for efm in efms]
efms_orig = efms[:100]

#b = Builder(map_name="RECON1.Carbohydrate metabolism", reaction_data=efms_orig[20125])
#b.save_html('/home/skapur/MEOCloud/Projectos/MCSEnumeratorPython/examples/GBconsensus_resources/EFMs/sample_efm.html')
#b.display_in_browser()
#b.reaction_data


class EFMTree(object):
	def __init__(self, value, extra_info=None):
		self.value = value
		self.children = []
		self.extra_info = extra_info

	def get_children(self):
		return [c for c in self.children]

	def add_child(self, node):
		self.children.append(node)

	def is_leaf(self):
		return self.children == []

	def __eq__(self, other):
		return self.value == other

	def __repr__(self):
		return str(self.value) + '(' + str(self.extra_info) + ')'

efm_sets = [set([r for r in efm.keys()]) for efm in efms_orig]

root = EFMTree('ROOT')

def fill_tree(tree, sets):
	if len(sets) > 0:
		counts = Counter(chain(*(sets)))
		if len(counts) > 0:
			most_common, max_value = counts.most_common(1)[0]
			#print(tab_level*"\t"+'Most common element is',most_common,'with',max_value,'hits.')
			new_node = EFMTree(most_common, extra_info=max_value)
			sets_containing_most_common = [setf for setf in sets if most_common in setf]
			#print(tab_level*"\t"+str(len(sets_containing_most_common)),'contain',most_common)
			remaining_sets = [setf for setf in sets if most_common not in setf]
			#print(tab_level*"\t"+str(len(remaining_sets)),"sets remaining.")
			tree.add_child(new_node)

			if len(sets_containing_most_common) > 0:
				fill_tree(new_node, [[k for k in setf if k != most_common] for setf in sets_containing_most_common])
			if len(remaining_sets) > 0:
				fill_tree(tree, remaining_sets)

def compress_linear_paths(tree):
	if tree.is_leaf():
		pass
	else:
		if len(tree.children) == 1:
			if type(tree.value) != list:
				tree.value = [tree.value]
			tree.value.append(tree.children[0].value)
			tree.children = tree.children[0].children
			compress_linear_paths(tree)
		for child in tree.children:
			compress_linear_paths(child)

fill_tree(root, efm_sets)
compress_linear_paths(root)

buffer = []
def pretty_print_tree(tree, buffer, overhead, final_node = True):
	current_line = overhead + "|-- "+repr(tree)
	buffer.append(current_line)
	for child in tree.children:
		pretty_print_tree(child, buffer, overhead+"|\t", False)
	if final_node:
		return buffer

pretty_print_tree(root, buffer, "", True)

with open('test_tree.txt', 'w') as f:
	f.write('\n'.join(buffer))

import networkx as nx

node_dict = {}
G = nx.Graph()


def populate_nx_graph(tree, G, previous=None):
	node_value_key = ' and '.join(tree.value) if type(tree.value) == list else str(tree.value)
	node_value = node_value_key
	if node_value_key not in node_dict.keys():
		node_dict[node_value_key] = 1
		node_value = node_value + "_" + '0'
	else:
		node_value = node_value + "_" + str(node_dict[node_value])
		node_dict[node_value_key] += 1

	if previous != None:
		previous_node, previous_key = previous
		G.add_edge(previous_node, node_value)
	if not tree.is_leaf():
		for child in tree.children:
			populate_nx_graph(child, G, previous=(node_value,node_value_key))


populate_nx_graph(root, G)
drawing = nx.draw(G)

tree_as_list = []

def populate_list_from_tree(tree):
	pass

import matplotlib.pyplot as plt