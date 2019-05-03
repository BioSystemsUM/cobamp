import pickle


def pickle_object(obj, path):
	"""
	Stores an object as a file.
	Parameters
	----------
	obj: The object instance
	path: Full path as a str where the file will be stored.

	Returns
	-------

	"""
	with open(path, "wb") as f:
		pickle.dump(obj, f)


def read_pickle(path):
	"""
	Reads a file containing a pickled object and returns it
	Parameters
	----------
	path: Full path as a str where the file is stored.

	Returns an object.
	-------

	"""
	with open(path, "rb") as f:
		return pickle.load(f)


def open_file(path, mode):
	with open(path, mode) as f:
		return f.read()
