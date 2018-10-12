import unittest

class COBRAWrapperTest(unittest.TestCase):
	def setUp(self):
		from cobra.io.sbml3 import read_sbml_model
		import metaconvexpy.connectors.cobrapy as connector

		model = read_sbml_model("/home/skapur/MEOCloud/Projectos/DeYeast/Models/iMM904/iMM904_peroxisome.xml")

		flux_space = {
			'EX_glc_e_' : (-1.15,-1.15),
			'EX_o2_e_' : (None, 0),
			'ATPM' : (8.39, 8.39)
		}
		yield_space = {
			('EX_succ_e_', 'EX_glc_e_') : (None, -0.001, None)
		}

		algorithm = connector.KShortestMCSEnumeratorWrapper(model, flux_space, yield_space)
		enumerator = algorithm.get_enumerator()
		solutions = next(enumerator)
		sol = solutions[0]
		sol.attribute_dict('signed_indicator_map')

if __name__ == '__main__':
	COBRAWrapperTest().setUp()