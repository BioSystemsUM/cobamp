import types


class PropertyDictionary():
	"""
	Implements a dict with additional control on which objects can be added to which keys and whether these are optional
	or mandatory.
	"""

	def __init__(self, mandatory_properties={}, optional_properties={}):

		"""
		The values for each of the required dicts can either be:
			- A type (such as str, int, etc...)
			- A function returning a boolean and accepting a single value as argument
			- A list of admissible values

		Parameters
		----------
		mandatory_properties: A dict[str,function] mapping the keys of mandatory properties with one of three options
		for values, as described above
		optional_properties: A dict[str,function] mapping the keys of optional properties with one of three options
		for values, as described above
		"""
		self.__mandatory_properties = mandatory_properties
		self.__optional_properties = optional_properties
		self.__properties = {}

	def add_new_properties(self, mandatory_properties, optional_properties):
		"""
		Adds new properties to the dictionary and/or updates existing ones, if present.
		Parameters
		----------
		mandatory_properties: A dict[str, function]
		optional_properties: A dict[str, function]
		-------

		"""
		self.__mandatory_properties.update(mandatory_properties)
		self.__optional_properties.update(optional_properties)

	def get_mandatory_properties(self):
		"""

		Returns a dictionary containing the mapping between mandatory keys and function/type/list controlling values.
		-------

		"""
		return self.__mandatory_properties

	def get_optional_properties(self):
		"""

		Returns a dictionary containing the mapping between optional keys and function/type/list controlling values.
		-------

		"""
		return self.__optional_properties

	def __getitem__(self, item):
		"""
		Overloaded indexing to allow the square brace syntax for accessing values through keys. If the key was not
		registered, an exception will be raised. If the key was registered but no value exists, None will be returned.

		Parameters
		----------
		item: Key for the value to be accessed

		Returns an object.
		-------

		"""
		if item not in self.__mandatory_properties.keys() and item not in self.__optional_properties.keys():
			raise Exception(str(item) + " has not been registered as a mandatory or optional property.")
		elif item not in self.__properties.keys():
			return None
		else:
			return self.__properties[item]

	def __setitem__(self, key, value):
		"""
		Sets the value for the supplied key
		Parameters
		----------
		key - str representing the key, preferrably contained in the mandatory or optional properties.
		value - an object compliant with the functions set for the key.
		-------

		"""
		if key in self.__mandatory_properties.keys() or key in self.__optional_properties.keys():
			expected_type = self.__mandatory_properties[key] if key in self.__mandatory_properties.keys() else \
				self.__optional_properties[key]
			if self.__check_key_value_pair(expected_type, value):
				self.__properties[key] = value
			else:
				raise Exception(str(key) + " does not accept the supplied `value` as a valid argument.")

	def has_required_properties(self):
		"""

		Returns a boolean value if the mandatory properties all have an associated value.
		-------

		"""
		return set(self.__properties.keys()) & set(self.__mandatory_properties.keys()) == set(
			self.__mandatory_properties.keys())

	def __check_key_value_pair(self, expected_type, value):
		"""
		Checks whether a value is compliant with a function/type or is contained in a list of admissible values.
		Parameters
		----------
		expected_type: A type to be compared with value, a function returning a boolean and accepting value as argument
		or a list of values where value should be contained.
		value

		Returns a boolean indicating whether the value can be added, assuming the conditions set by `expected_type`
		-------

		"""
		if type(expected_type) is type:
			is_ok = expected_type == type(value)
			if not is_ok:
				raise TypeError(
					"\'value\' has type " + str(type(value)) + " but \'key\' requires type " + str(expected_type))
		elif type(expected_type) == types.FunctionType:
			is_ok = expected_type(value)
			if not is_ok:
				raise AssertionError(
					"Property checking function " + expected_type.__name__ + " does not allow the specified \'value\'")
		else:
			is_ok = value in expected_type
			if not is_ok:
				raise AssertionError("\'value\' is not contained in " + str(expected_type))

		return is_ok

	def add_if_not_none(self, key, value):
		if value is not None:
			self[key] = value

	def __repr__(self):
		"""
		Returns a string representation of the internal dictionary where all keys/values are stored.
		-------

		"""
		return '\n'.join([str(k) + " = " + str(v) for k, v in self.__properties.items()])
