import unittest
from metaconvexpy.utilities.property_management import PropertyDictionary
class PropertyDictionaryTest(unittest.TestCase):
	def setUp(self):
		base_mandatory = {'name':str, 'age':int, 'height':lambda x: x > 0, 'gender':['M','F','U']}
		base_optional = {'car_brand':str, 'region':['rural','urban'], 'tyre_sizes':lambda x: len(x) == 2}

		class CustomPropertyDictionary(PropertyDictionary):
			def __init__(self):
				super().__init__(base_mandatory, base_optional)

		self.dict_class = CustomPropertyDictionary
	def test_add_all_mandatory_info(self):
		propdict = self.dict_class()
		propdict['name'] = 'John'
		propdict['age'] = 29
		propdict['height'] = 2
		propdict['gender'] = 'M'
		propdict['car_brand'] = 'Skoda'

		keys_to_check = ['name', 'age', 'height', 'gender', 'car_brand']
		values_to_check = ('John', 29, 2, 'M', 'Skoda')
		propdict_is_valid = propdict.has_required_properties()
		propdict_has_added_keys = tuple(propdict[key] for key in keys_to_check)

		self.assertTrue(propdict_is_valid)
		self.assertTrue(values_to_check == propdict_has_added_keys)

	def test_add_some_mandatory_info(self):
		propdict = self.dict_class()
		propdict['name'] = 'John'
		propdict['age'] = 29
		propdict['gender'] = 'M'
		propdict['car_brand'] = 'Skoda'

		keys_to_check = ['name', 'age', 'gender', 'car_brand']
		values_to_check = ('John', 29, 'M', 'Skoda')

		propdict_is_valid = propdict.has_required_properties()
		propdict_has_added_keys = tuple(propdict[key] for key in keys_to_check)

		self.assertTrue(not propdict_is_valid)
		self.assertTrue(values_to_check == propdict_has_added_keys)

	def test_add_info_of_wrong_type(self):
		with self.assertRaises(TypeError) as context:
			propdict = self.dict_class()
			propdict['name'] = 'John'
			propdict['age'] = lambda x: x
		propdict_is_valid = propdict.has_required_properties()
		self.assertTrue(not propdict_is_valid)


	def test_add_info_noncompliant_with_function(self):
		with self.assertRaises(AssertionError) as context:
			propdict = self.dict_class()
			propdict['name'] = 'John'
			propdict['height'] = 0
		propdict_is_valid = propdict.has_required_properties()
		self.assertTrue(not propdict_is_valid)


	def test_add_info_not_in_list(self):
		with self.assertRaises(AssertionError) as context:
			propdict = self.dict_class()
			propdict['name'] = 'John'
			propdict['gender'] = 'X'
		propdict_is_valid = propdict.has_required_properties()
		self.assertTrue(not propdict_is_valid)

if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(PropertyDictionaryTest)
	unittest.TextTestRunner(verbosity=2).run(suite)

