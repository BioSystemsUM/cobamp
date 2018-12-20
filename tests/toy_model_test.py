from cobamp.algorithms.kshortest import KShortestEnumerator
from cobamp.core.linear_systems import DualLinearSystem, IrreversibleLinearSystem
import numpy as np
from itertools import chain
import unittest


class ToyMetabolicNetworkTests(unittest.TestCase):
	def setUp(self):
		self.S = np.array([[1, -1, 0, 0, -1, 0, -1, 0, 0],
		                   [0, 1, -1, 0, 0, 0, 0, 0, 0],
		                   [0, 1, 0, 1, -1, 0, 0, 0, 0],
		                   [0, 0, 0, 0, 0, 1, -1, 0, 0],
		                   [0, 0, 0, 0, 0, 0, 1, -1, 0],
		                   [0, 0, 0, 0, 1, 0, 0, 1, -1]])
		self.rx_names = ["R" + str(i) for i in range(1, 10)]
		self.irrev = [0, 1, 2, 4, 5, 6, 7, 8]

		self.T = np.array([0] * self.S.shape[1]).reshape(1, self.S.shape[1])
		self.T[0, 8] = -1
		self.b = np.array([-1]).reshape(1, )

	def enumerate_elementary_flux_modes(self):
		lsystem = IrreversibleLinearSystem(self.S, self.irrev)

		ksh = KShortestEnumerator(lsystem)
		solution_iterator = ksh.population_iterator(9)
		efms = list(chain(*solution_iterator))
		return efms

	def enumerate_minimal_cut_sets(self):
		dsystem = DualLinearSystem(self.S, self.irrev, self.T, self.b)

		ksh = KShortestEnumerator(dsystem)
		solution_iterator = ksh.population_iterator(4)
		mcss = list(chain(*solution_iterator))
		return mcss

	def test_elementary_flux_modes_support(self):
		basic_answer = {"R1, R2, R3, R4", "R1, R4, R5, R9", "R1, R2, R3, R5, R9", "R1, R6, R7, R8, R9"}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_elementary_flux_modes()}
		self.assertEqual(basic_answer, test)

	# def test_elementary_flux_modes_distribution(self):
	#
	# 	test = {self.convert_solution_to_string(sol) for sol in self.enumerate_elementary_flux_modes()}
	# 	sol.
	# 	self.assertEqual(basic_answer, test)

	def test_minimal_cut_sets(self):
		answer = {'R1', 'R2, R4, R6', 'R2, R4, R7', 'R2, R4, R8', 'R3, R4, R6', 'R3, R4, R7', 'R3, R4, R8', 'R5, R6',
		          'R5, R7', 'R5, R8', 'R9'}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_minimal_cut_sets()}
		self.assertEqual(answer, test)

	def convert_solution_to_string(self, sol):
		return ', '.join([self.rx_names[i] for i in sol.get_active_indicator_varids()])


if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(ToyMetabolicNetworkTests)
	unittest.TextTestRunner(verbosity=2).run(suite)
