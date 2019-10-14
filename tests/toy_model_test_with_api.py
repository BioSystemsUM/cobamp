from cobamp.algorithms.kshortest import *
from cobamp.core.linear_systems import DualLinearSystem, IrreversibleLinearSystem
import numpy as np
import unittest


efm_populate_enumeration_config = KShortestProperties()
efm_populate_enumeration_config[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_POPULATE
efm_populate_enumeration_config[K_SHORTEST_OPROPERTY_MAXSIZE] = 9

mcs_populate_enumeration_config = KShortestProperties()
mcs_populate_enumeration_config[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_POPULATE
mcs_populate_enumeration_config[K_SHORTEST_OPROPERTY_MAXSIZE] = 3

efm_populate_enumeration_config_wrong = KShortestProperties()
efm_populate_enumeration_config_wrong[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_POPULATE
efm_populate_enumeration_config_wrong[K_SHORTEST_OPROPERTY_MAXSIZE] = 4

mcs_populate_enumeration_config_wrong = KShortestProperties()
mcs_populate_enumeration_config_wrong[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_POPULATE
mcs_populate_enumeration_config_wrong[K_SHORTEST_OPROPERTY_MAXSIZE] = 2

efm_iterate_enumeration_config = KShortestProperties()
efm_iterate_enumeration_config[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_ITERATE
efm_iterate_enumeration_config[K_SHORTEST_OPROPERTY_MAXSOLUTIONS] = 4

mcs_iterate_enumeration_config = KShortestProperties()
mcs_iterate_enumeration_config[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_ITERATE
mcs_iterate_enumeration_config[K_SHORTEST_OPROPERTY_MAXSOLUTIONS] = 11

efm_iterate_enumeration_config_wrong = KShortestProperties()
efm_iterate_enumeration_config_wrong[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_ITERATE
efm_iterate_enumeration_config_wrong[K_SHORTEST_OPROPERTY_MAXSOLUTIONS] = 2

mcs_iterate_enumeration_config_wrong = KShortestProperties()
mcs_iterate_enumeration_config_wrong[K_SHORTEST_MPROPERTY_METHOD] = K_SHORTEST_METHOD_ITERATE
mcs_iterate_enumeration_config_wrong[K_SHORTEST_OPROPERTY_MAXSOLUTIONS] = 9

configs = [efm_populate_enumeration_config, mcs_populate_enumeration_config, efm_populate_enumeration_config_wrong, mcs_populate_enumeration_config_wrong,
 efm_iterate_enumeration_config, mcs_iterate_enumeration_config, efm_iterate_enumeration_config_wrong, mcs_iterate_enumeration_config_wrong]

for cfg in configs:
	cfg[K_SHORTEST_OPROPERTY_BIG_M_VALUE] = 3.4200101010 * 1e4
	cfg[K_SHORTEST_MPROPERTY_TYPE_EFP] = False
	cfg[K_SHORTEST_OPROPERTY_N_THREADS] = 1
	cfg[K_SHORTEST_OPROPERTY_WORKMEMORY] = None

TEST_SOLVER = 'CPLEX'

class ToyMetabolicNetworkTests(unittest.TestCase):
	def setUp(self):
		self.S = np.array([[1, -1, 0, 0, -1, 0, -1, 0, 0],
		                   [0, 1, -1, 0, 0, 0, 0, 0, 0],
		                   [0, 1, 0, 1, -1, 0, 0, 0, 0],
		                   [0, 0, 0, 0, 0, 1, -1, 0, 0],
		                   [0, 0, 0, 0, 0, 0, 1, -1, 0],
		                   [0, 0, 0, 0, 1, 0, 0, 1, -1]])
		self.rx_names = ["R" + str(i) for i in range(1, 10)]
		self.lb, self.ub = [0]*len(self.rx_names), [1000]*len(self.rx_names)
		self.lb[3] = -1000
		self.T = np.array([0] * self.S.shape[1]).reshape(1, self.S.shape[1])
		self.T[0, 8] = -1
		self.b = np.array([-1]).reshape(1, )
		self.lsystem = IrreversibleLinearSystem(self.S, self.lb, self.ub, solver=TEST_SOLVER)
		self.dsystem = DualLinearSystem(self.S, self.lb, self.ub, self.T, self.b, solver=TEST_SOLVER)



	def enumerate_elementary_flux_modes(self):
		ks = KShortestEFMAlgorithm(efm_populate_enumeration_config)
		r =  ks.enumerate(self.lsystem)
		#print('Thread_parameter',ks.ksh.model.model.problem.parameters.workmem.get())
		return r

	def enumerate_some_elementary_flux_modes(self):
		ks = KShortestEFMAlgorithm(efm_populate_enumeration_config_wrong)
		r =  ks.enumerate(self.lsystem)
		#print('Thread_parameter',ks.ksh.model.model.problem.parameters.threads.get())
		return r

	def enumerate_minimal_cut_sets(self):
		ks = KShortestEFMAlgorithm(mcs_populate_enumeration_config)
		r = ks.enumerate(self.dsystem)
		#print('Thread_parameter', ks.ksh.model.model.problem.parameters.threads.get())
		return r

	def enumerate_some_minimal_cut_sets(self):
		ks = KShortestEFMAlgorithm(mcs_populate_enumeration_config_wrong)
		r = ks.enumerate(self.dsystem)
		#print('Thread_parameter', ks.ksh.model.model.problem.parameters.threads.get())
		return r

	def enumerate_elementary_flux_modes_iter(self):
		ks = KShortestEFMAlgorithm(efm_iterate_enumeration_config)
		r = ks.enumerate(self.lsystem)
		#print('Thread_parameter', ks.ksh.model.model.problem.parameters.threads.get())
		return r

	def enumerate_some_elementary_flux_modes_iter(self):
		ks = KShortestEFMAlgorithm(efm_iterate_enumeration_config_wrong)
		r = ks.enumerate(self.lsystem)
		#print('Thread_parameter', ks.ksh.model.model.problem.parameters.threads.get())
		return r

	def enumerate_minimal_cut_sets_iter(self):
		ks = KShortestEFMAlgorithm(mcs_iterate_enumeration_config)
		r = ks.enumerate(self.dsystem)
		#print('Thread_parameter', ks.ksh.model.model.problem.parameters.threads.get())
		return r

	def enumerate_some_minimal_cut_sets_iter(self):
		ks = KShortestEFMAlgorithm(mcs_iterate_enumeration_config_wrong)
		r = ks.enumerate(self.dsystem)
		#print('Thread_parameter', ks.ksh.model.model.problem.parameters.threads.get())
		return r

	def test_elementary_flux_modes_support(self):
		basic_answer = {"R1, R2, R3, R4", "R1, R4, R5, R9", "R1, R2, R3, R5, R9", "R1, R6, R7, R8, R9"}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_elementary_flux_modes()}
		self.assertEqual(basic_answer, test)

	def test_elementary_flux_modes_support_wrong(self):
		basic_answer = {"R1, R2, R3, R4", "R1, R4, R5, R9", "R1, R2, R3, R5, R9", "R1, R6, R7, R8, R9"}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_some_elementary_flux_modes()}
		self.assertNotEqual(basic_answer, test)

	def test_minimal_cut_sets(self):
		answer = {'R1', 'R2, R4, R6', 'R2, R4, R7', 'R2, R4, R8', 'R3, R4, R6', 'R3, R4, R7', 'R3, R4, R8', 'R5, R6',
		          'R5, R7', 'R5, R8', 'R9'}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_minimal_cut_sets()}
		self.assertEqual(answer, test)

	def test_minimal_cut_sets_wrong(self):
		answer = {'R1', 'R2, R4, R6', 'R2, R4, R7', 'R2, R4, R8', 'R3, R4, R6', 'R3, R4, R7', 'R3, R4, R8', 'R5, R6',
		          'R5, R7', 'R5, R8', 'R9'}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_some_minimal_cut_sets()}
		self.assertNotEqual(answer, test)
		
	def test_elementary_flux_modes_support_iter(self):
		basic_answer = {"R1, R2, R3, R4", "R1, R4, R5, R9", "R1, R2, R3, R5, R9", "R1, R6, R7, R8, R9"}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_elementary_flux_modes_iter()}
		self.assertEqual(basic_answer, test)

	def test_elementary_flux_modes_support_wrong_iter(self):
		basic_answer = {"R1, R2, R3, R4", "R1, R4, R5, R9", "R1, R2, R3, R5, R9", "R1, R6, R7, R8, R9"}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_some_elementary_flux_modes_iter()}
		self.assertNotEqual(basic_answer, test)

	def test_minimal_cut_sets_iter(self):
		answer = {'R1', 'R2, R4, R6', 'R2, R4, R7', 'R2, R4, R8', 'R3, R4, R6', 'R3, R4, R7', 'R3, R4, R8', 'R5, R6',
		          'R5, R7', 'R5, R8', 'R9'}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_minimal_cut_sets_iter()}
		self.assertEqual(answer, test)

	def test_minimal_cut_sets_wrong_iter(self):
		answer = {'R1', 'R2, R4, R6', 'R2, R4, R7', 'R2, R4, R8', 'R3, R4, R6', 'R3, R4, R7', 'R3, R4, R8', 'R5, R6',
		          'R5, R7', 'R5, R8', 'R9'}
		test = {self.convert_solution_to_string(sol) for sol in self.enumerate_some_minimal_cut_sets_iter()}
		self.assertNotEqual(answer, test)

	def convert_solution_to_string(self, sol):
		return ', '.join([self.rx_names[i] for i in sol.get_active_indicator_varids()])


if __name__ == '__main__':
	suite = unittest.TestLoader().loadTestsFromTestCase(ToyMetabolicNetworkTests)
	unittest.TextTestRunner(verbosity=2).run(suite)
