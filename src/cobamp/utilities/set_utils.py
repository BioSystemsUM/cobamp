def is_subset(which, of_what):
	"""

	Parameters
	----------
	which: A set or frozenset instance to be checked as a possible subset of `of_what`
	of_what: A set or frozenset instance

	Returns a boolean indicating if `which` is a subset of `of_what`
	-------

	"""
	return 0 < len(which & of_what) <= len(of_what)


def is_identical(set1, set2):
	"""

	Parameters
	----------
	set1: A set of frozenset instance.
	set2: A set of frozenset instance.

	Returns a boolean indicating if both set1 and set2 are identical (contain exactly the same elements)
	-------

	"""
	return len(set1 & set2) == len(set1) == len(set2)


def has_no_overlap(set1, set2):
	"""

	Parameters
	----------
	set1: A set of frozenset instance.
	set2: A set of frozenset instance.

	Returns a boolean indicating if the intersection of set1 and set2 is empty.
	-------

	"""
	return len(set1 & set2) == 0
