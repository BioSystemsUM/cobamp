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

