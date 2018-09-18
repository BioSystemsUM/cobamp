from itertools import chain
from collections import Counter
import networkx as nx

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

def ignore_compressed_nodes_by_size(tree, size):
	for child in tree.children:
		if isinstance(child.value, list) and len(child.value) > size:
			child.value = 'REMAINING'
			child.children = []
		else:
			ignore_compressed_nodes_by_size(child, size)

def probabilistic_tree_prune(tree, target_level, current_level=0, cut_leaves = False, name_separator=' and '):
	if current_level < target_level:
		if target_level == current_level and cut_leaves:
			tree.children = []
		else:
			for child in tree.children:
				probabilistic_tree_prune(child, target_level, current_level+1, cut_leaves, name_separator)

	else:
		probabilistic_tree_compression(tree, name_separator=name_separator)
		tree.value = "REMAINING" if cut_leaves else [str(k)+"="+str(tree.value[k]) for k in sorted(tree.value, key=tree.value.get)]
	return target_level == current_level

def probabilistic_tree_compression(tree, data=None, total_count=None, name_separator=' and '):
	if data is None and total_count is None:
		total_count = int(tree.extra_info)
		data = {name_separator.join(tree.value) if isinstance(tree.value, list) else tree.value: 1}
		for child in tree.children:
			probabilistic_tree_compression(child, data, total_count, name_separator)
		tree.value = data
		tree.children = []
	else:
		local_proportion = int(tree.extra_info)/total_count
		key = name_separator.join(tree.value) if isinstance(tree.value, list) else tree.value
		if key not in data.keys():
			data[key] = local_proportion
		else:
			data[key] += local_proportion
		for child in tree.children:
			probabilistic_tree_compression(child, data, total_count, name_separator)


def pretty_print_tree(tree, write_path=None):
	buffer = []

	def __pretty_print_tree(tree, buffer, overhead, final_node=True):
		current_line = overhead + "|-- " + repr(tree)
		buffer.append(current_line)
		for child in tree.children:
			__pretty_print_tree(child, buffer, overhead + "|\t", False)
		if final_node:
			return buffer

	__pretty_print_tree(tree, buffer, '', True)

	res = '\n'.join(buffer)

	if write_path is not None:
		with open('test_tree.txt', 'w') as f:
			f.write(res)

	return res

def apply_fx_to_all_node_values(tree, fx):
	tree.value = fx(tree.value)
	for child in tree.children:
		apply_fx_to_all_node_values(child, fx)

def find_all_tree_nodes(tree):
	def __find_all_tree_nodes(tree, it):
		it.append(tree)
		for child in tree.children:
			__find_all_tree_nodes(child, it)

	it = []
	__find_all_tree_nodes(tree, it)
	return it


def merge_duplicate_nodes(tree):
	all_nodes = find_all_tree_nodes(tree)
	conv_key = lambda x: str(sorted(x) if isinstance(x, list) else x)
	unique_keys = [conv_key(k.value) for k in all_nodes]
	unique_node_map = {k:EFMTree(k) for k in unique_keys}

	def __merge_duplicate_nodes(tree):
		new_children = []
		for child in tree.children:
			grandchildren = child.children
			new_child = unique_node_map[conv_key(child.value)]
			new_child.children = grandchildren
			new_children.append(new_child)
		tree.children = new_children
		for child in tree.children:
			__merge_duplicate_nodes(child)
	__merge_duplicate_nodes(tree)


def populate_nx_graph(tree, G, previous=None, name_separator='\n', unique_nodes=True, node_dict=None):
	if node_dict is None:
		node_dict = {}
	if unique_nodes:
		node_value_key = name_separator.join(tree.value) if type(tree.value) == list else str(tree.value)
		node_value = node_value_key
		if node_value_key not in node_dict.keys():
			node_dict[node_value_key] = 1
			node_value = node_value + "_" + '0'
		else:
			node_value = node_value + "_" + str(node_dict[node_value])
			node_dict[node_value_key] += 1
	else:
		node_value = tree.value
		node_value_key = tree.value
	if previous != None:
		previous_node, previous_key = previous
		G.add_edge(previous_node, node_value)
	if not tree.is_leaf():
		for child in tree.children:
			populate_nx_graph(child, G, previous=(node_value,node_value_key), name_separator=name_separator, unique_nodes=unique_nodes, node_dict=node_dict)
