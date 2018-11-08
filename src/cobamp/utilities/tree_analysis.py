from itertools import chain
from collections import Counter


class Tree(object):
	"""
	A simple class representing an n-ary tree as a node with one or more children nodes.
	"""

	def __init__(self, value, extra_info=None):
		"""
		Initializes the tree with no children.
		Parameters
		----------
		value: A value associated with this node
		extra_info: Additional hidden information present in the node (OPTIONAL)
		"""
		self.value = value
		self.children = []
		self.extra_info = extra_info

	def get_children(self):
		"""

		Returns a list with all the children of this node.
		-------

		"""
		return [c for c in self.children]

	def add_child(self, node):
		"""
		Adds a node to the list of children in this node.
		Parameters
		----------
		node: A <Tree> instance
		-------

		"""
		self.children.append(node)

	def is_leaf(self):
		"""
		Checks whether this node has no children (is a leaf)
		Returns a boolean.
		-------

		"""
		return self.children == []

	def __eq__(self, other):
		"""
		Overloaded equality comparison operator. Compares the value of both nodes.
		Parameters
		----------
		other

		Returns a boolean
		-------

		"""
		return self.value == other

	def __repr__(self):
		"""

		Returns a representation of this node as a string containing the value and extra information.
		-------

		"""
		return str(self.value) + '(' + str(self.extra_info) + ')'


def fill_tree(tree, sets):
	"""
	Fills a Tree instance with data from the Iterable[set/frozenset] supplied as the argument `sets`.
	The resulting tree will be filled in a way that each set can be retrieved by traversing from the root node `tree`
	towards a leaf. The nodes required to travel down this path contain the values of a single set.
	The resulting tree will not contain circular references so this should not be treated as a graph. The filling method
	is recursive, so each child will be filled with sets contained in the parent node. Sets that have already been added
	are removed from the original pool.
	Elements to be added as nodes are chosen by the frequency at which they occur in the sets.
	Parameters
	----------
	tree: A tree instance
	sets: A list of set/frozenset instances.
	-------

	"""
	if len(sets) > 0:
		counts = Counter(chain(*(sets)))
		if len(counts) > 0:
			most_common, max_value = counts.most_common(1)[0]
			# print(tab_level*"\t"+'Most common element is',most_common,'with',max_value,'hits.')
			new_node = Tree(most_common, extra_info=max_value)
			sets_containing_most_common = [setf for setf in sets if most_common in setf]
			# print(tab_level*"\t"+str(len(sets_containing_most_common)),'contain',most_common)
			remaining_sets = [setf for setf in sets if most_common not in setf]
			# print(tab_level*"\t"+str(len(remaining_sets)),"sets remaining.")
			tree.add_child(new_node)

			if len(sets_containing_most_common) > 0:
				fill_tree(new_node, [[k for k in setf if k != most_common] for setf in sets_containing_most_common])
			if len(remaining_sets) > 0:
				fill_tree(tree, remaining_sets)


def compress_linear_paths(tree):
	"""
	Collapses sequences of nodes contained in a Tree with only one children as a single node containing all values of
	those nodes.
	Parameters
	----------
	tree: A Tree instance.
	-------

	"""
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
	"""
	Modifies the values of a tree's children that have been previously compressed with the <compress_linear_paths>
	function if they contain more than a certain number of elements. The node's value is changed to "REMAINING".
	Parameters
	----------
	tree: A Tree instance
	size: An integer with the size threshold
	-------

	"""
	for child in tree.children:
		if isinstance(child.value, list) and len(child.value) > size:
			child.value = 'REMAINING'
			child.children = []
		else:
			ignore_compressed_nodes_by_size(child, size)


def probabilistic_tree_prune(tree, target_level, current_level=0, cut_leaves=False, name_separator=' and '):
	"""
	Cuts a tree's nodes under a certain height (`target_level`) and converts ensuing nodes into a single one whose value
	represents the relative frequency of an element in the nodes below. Requires values on the extra_info field.
	Parameters
	----------
	tree: A Tree instance
	target_level: An int representing the level at which the tree will be cut
	current_level: The current level of the tree (int). Default is 0 for root nodes.
	cut_leaves: A boolean indicating whether the node at the target level is excluded or displays probabilities.
	name_separator: Separator to use when representing multiple elements
	-------

	"""
	if current_level < target_level:
		if target_level == current_level and cut_leaves:
			tree.children = []
		else:
			for child in tree.children:
				probabilistic_tree_prune(child, target_level, current_level + 1, cut_leaves, name_separator)

	else:
		probabilistic_tree_compression(tree, name_separator=name_separator)
		tree.value = "REMAINING" if cut_leaves else [str(k) + "=" + str(tree.value[k]) for k in
													 sorted(tree.value, key=tree.value.get)]
	return target_level == current_level


def probabilistic_tree_compression(tree, data=None, total_count=None, name_separator=' and '):
	"""
	Compresses a node and subsequent children by removing them and modifying the value to a dictionary with the relative
	frequency of each element in the subsequent nodes. Requires values on the extra_info field.

	Parameters
	----------
	tree: A Tree instance
	data: Local count if not available in extra_info
	total_count: Total amount of sets if not available in extra_info
	name_separator: Separator to use when representing multiple elements
	-------

	"""
	if data is None and total_count is None:
		total_count = int(tree.extra_info)
		data = {name_separator.join(tree.value) if isinstance(tree.value, list) else tree.value: 1}
		for child in tree.children:
			probabilistic_tree_compression(child, data, total_count, name_separator)
		tree.value = data
		tree.children = []
	else:
		local_proportion = int(tree.extra_info) / total_count
		key = name_separator.join(tree.value) if isinstance(tree.value, list) else tree.value
		if key not in data.keys():
			data[key] = local_proportion
		else:
			data[key] += local_proportion
		for child in tree.children:
			probabilistic_tree_compression(child, data, total_count, name_separator)


def pretty_print_tree(tree, write_path=None):
	"""
	Parameters
	----------
	tree: A Tree instance
	write_path: Path to store a text file. Use None if the string is not to be stored in a file.

	Returns a text representation of a Tree instance along with its children.
	-------

	"""
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
	"""
	Applies a function to all nodes below the tree, modifying their value to its result.
	Parameters
	----------
	tree: A Tree instance
	fx: A function to apply
	-------

	"""
	tree.value = fx(tree.value)
	for child in tree.children:
		apply_fx_to_all_node_values(child, fx)


def find_all_tree_nodes(tree):
	"""
	Parameters
	----------
	tree: A Tree instance.

	Returns a list of all nodes below a node
	-------

	"""

	def __find_all_tree_nodes(tree, it):
		it.append(tree)
		for child in tree.children:
			__find_all_tree_nodes(child, it)

	it = []
	__find_all_tree_nodes(tree, it)
	return it


def merge_duplicate_nodes(tree):
	"""
	Merges all nodes with similar values, replacing every instance reference of all nodes with the same object if its
	value is identical

	Parameters
	----------
	tree: A Tree instance
	-------

	"""
	all_nodes = find_all_tree_nodes(tree)
	conv_key = lambda x: str(sorted(x) if isinstance(x, list) else x)
	unique_keys = [conv_key(k.value) for k in all_nodes]
	unique_node_map = {k: Tree(k) for k in unique_keys}

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
			populate_nx_graph(child, G, previous=(node_value, node_value_key), name_separator=name_separator,
							  unique_nodes=unique_nodes, node_dict=node_dict)
