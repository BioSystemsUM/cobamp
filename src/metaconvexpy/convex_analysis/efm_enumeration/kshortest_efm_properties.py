from ...utilities.property_management import PropertyDictionary

K_SHORTEST_MPROPERTY_METHOD = 'METHOD'
K_SHORTEST_METHOD_ITERATE = "ITERATE"
K_SHORTEST_METHOD_POPULATE = "POPULATE"

K_SHORTEST_OPROPERTY_MAXSIZE = 'MAXSIZE'
K_SHORTEST_OPROPERTY_MAXSOLUTIONS = "MAXSOLUTIONS"

kshortest_mandatory_properties = {
	K_SHORTEST_MPROPERTY_METHOD: [K_SHORTEST_METHOD_ITERATE,K_SHORTEST_METHOD_POPULATE]}

kshortest_optional_properties = {
	K_SHORTEST_OPROPERTY_MAXSIZE: lambda x: x > 0 and isinstance(x, int),
	K_SHORTEST_OPROPERTY_MAXSOLUTIONS: lambda x: x > 0 and isinstance(x, int)
}

class KShortestProperties(PropertyDictionary):
	'''
	Class defining a configuration for the K-shortest algorithm.
	The following fields are mandatory:
	K_SHORTEST_MPROPERTY_METHOD:
		- K_SHORTEST_METHOD_ITERATE : Iterative enumeration (one EFM at a time)
		- K_SHORTEST_METHOD_POPULATE : Enumeration by size (EFMs of a certain size at a time)
	'''
	def __init__(self):
		super().__init__(kshortest_mandatory_properties, kshortest_optional_properties)
