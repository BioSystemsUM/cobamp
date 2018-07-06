from metaconvexpy.utilities.property_management import PropertyDictionary

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
	def __init__(self):
		super().__init__(kshortest_mandatory_properties, kshortest_optional_properties)
