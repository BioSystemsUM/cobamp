import types

class PropertyDictionary(object):
	def __init__(self, mandatory_properties, optional_properties):
		self.__properties = {}
		self.__mandatory_properties = mandatory_properties
		self.__optional_properties = optional_properties

	def __getitem__(self, item):
		if item not in self.__mandatory_properties.keys() and item not in self.__optional_properties.keys():
			raise Exception(str(item)+" has not been registered as a mandatory or optional property.")
		else:
			return self.__properties[item]

	def __setitem__(self, key, value):
		if key in self.__mandatory_properties.keys() or key in self.__optional_properties.keys():
			expected_type = self.__mandatory_properties[key] if key in self.__mandatory_properties.keys() else self.__optional_properties[key]
			if self.__check_key_value_pair(expected_type, value):
				self.__properties[key] = value

	def has_required_properties(self):
		return set(self.__properties) & set(self.__mandatory_properties.keys()) == set(self.__mandatory_properties.keys())

	def __check_key_value_pair(self, expected_type, value):
		if type(expected_type) is type:
			is_ok = expected_type == type(value)
			if not is_ok:
				raise TypeError("\'value\' has type " + str(type(value)) + " but \'key\' requires type " + str(expected_type))
		elif type(expected_type) == types.FunctionType:
			is_ok = expected_type(value)
			if not is_ok:
				raise AssertionError("Property checking function "+expected_type.__name__+" does not allow the specified \'value\'")
		else:
			is_ok = value in expected_type
			if not is_ok:
				raise AssertionError("\'value\' is not contained in "+str(expected_type))

		return is_ok