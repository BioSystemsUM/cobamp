import unittest
from scipy.sparse import csc_matrix
from numpy import array
from cobamp.core.models import ConstraintBasedModel
from cobamp.wrappers import KShortestMCSEnumeratorWrapper, KShortestGeneticMCSEnumeratorWrapper
from cobamp.gpr.evaluator import GPREvaluator
from cobamp.gpr.integration import GeneMatrixBuilder
from itertools import chain

class GMCSToyModelTest(unittest.TestCase):
	def setUp(self) -> None:
		rows, cols, data = zip(*(
			(0, 0, 1),
			(0, 1, -1),
			(0, 2, -1),
			(1, 1, 1),
			(1, 3, -1),
			(2, 3, 1),
			(2, 5, -1),
			(3, 5, 1),
			(3, 7, -1),
			(4, 2, 1),
			(4, 4, -1),
			(5, 4, 1),
			(5, 6, -1),
			(3, 6, 1)
		))

		# G_test = array([
		# 	[1,0,0,0,0,0,0,0,0,0],
		# 	[0,1,0,1,0,1,0,1,0,0],
		# 	[0,0,1,0,0,0,0,0,0,0],
		# 	[0,0,1,0,0,0,0,0,0,0],
		# 	[0,0,0,1,0,0,0,0,0,0],
		# 	[0,0,1,0,0,0,1,0,1,0],
		# 	[0,1,0,1,1,1,0,1,0,0],
		# 	[0,0,0,1,1,0,0,0,0,0]
		# ])
		#
		# F_test = array([
		# 	[1,0,0,0,0,0,0],
		# 	[0,1,0,0,0,0,0],
		# 	[0,0,1,0,0,0,0],
		# 	[0,0,0,1,0,0,0],
		# 	[0,0,0,0,1,0,0],
		# 	[0,0,1,0,0,1,0],
		# 	[0,1,0,0,0,0,1],
		# 	[0,0,0,0,1,1,1]
		# ])
		#
		# G_test_irrev = G_test[:,[0,5,1,2,3,6,4,9,7,8]]

		S = array(csc_matrix((data, (rows, cols))).todense())
		lb = array([0] * S.shape[1]).astype(float)
		ub = array([1000] * S.shape[1]).astype(float)
		lb[[1, 5]] = -1000
		rx_names = ['r' + str(i + 1) for i in range(S.shape[1] - 1)] + ['rbio']
		met_names = ['m' + str(i + 1) for i in range(S.shape[0] - 1)] + ['mbio']
		gprs = ['g1', 'g2', 'g2', 'g3 and g4', 'g2 and g5', 'g3 or g6', '(g2 and (g5 or g6)) or g7', '']
		gprs_irrev = gprs + [g for i, g in enumerate(gprs) if i in [1, 5]]

		#
		gmat_builder = GeneMatrixBuilder(GPREvaluator(gprs_irrev))
		G_new, F_new, gene_map_new = gmat_builder.get_GF_matrices()

		# G,F,gene_map = G_test_irrev, F_test, {'g'+str(i+1):i for i in range(F_test.shape[1])}
		cbm = ConstraintBasedModel(S, list(zip(lb, ub)), reaction_names=rx_names, metabolite_names=met_names)
		irrev_cbm, mapping = cbm.make_irreversible()
		irreducible_gene_map = {' and '.join({v: k for k, v in gene_map_new.items()}[k] for k in row.nonzero()[0]): i
								for i, row in enumerate(F_new)}

		gene_sets = [set(f.nonzero()[0]) for f in F_new]
		weights = [len(g) for g in gene_sets]
		rev_map = {v: k for k, v in irreducible_gene_map.items()}
		F_deps = {i: {j for j, h in enumerate(gene_sets) if len(h & g) == len(g) and i != j} for i, g in
				  enumerate(gene_sets)}
		gmcs_enumerator = KShortestGeneticMCSEnumeratorWrapper(
			model=irrev_cbm,
			target_flux_space_dict={'rbio': (1, None)},
			target_yield_space_dict={},
			stop_criteria=len(irrev_cbm.reaction_names),
			algorithm_type='kse_populate',
			excluded_solutions=[],
			G=G_new, gene_map=irreducible_gene_map, F=F_deps, gene_weights=weights
		)

		self.gmcs_enumerator = gmcs_enumerator

	def test_enumerator_results_toy_model(self):
		iterator = self.gmcs_enumerator.get_enumerator()
		solutions = set([frozenset(s.keys()) for s in list(chain(*iterator))])
		expected = set([frozenset(k) for k in [{'g1'},{'g2'},{'g3','g5'},{'g4','g5'}]])
		self.assertEqual(solutions, expected)

if __name__ == '__main__':
	unittest.main()
