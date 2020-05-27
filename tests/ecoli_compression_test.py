
from itertools import chain, product

from cobamp.nullspace.subset_reduction import SubsetReducerProperties, SubsetReducer
from cobamp.core.cb_analysis import FluxVariabilityAnalysis
from cobamp.wrappers import COBRAModelObjectReader, KShortestMCSEnumeratorWrapper
from cobamp.utilities.file_io import open_file

import unittest

class MCSEnumeratorQuickTest(unittest.TestCase):
	EXPECTED_COMP_SHAPE = 562,936

	def setUp(self):
		# read model using cobra (alternatively, the model instance can be generated previously and passed as an instance)
		reader = COBRAModelObjectReader('resources/iaf1260/Ec_iAF1260_flux2.xml')

		# drains/transporters/pseudo-reactions to exclude from network compression
		self.singles = [k+'_' if k[:3] == 'EX_' else k for k in [s[2:].strip() for s in
			open_file('resources/iaf1260/iAF1260_comp_exclusions.txt','r').split('\n')]]

		# generate a ConstraintBasedModel instance with an optimizer
		self.cbm = reader.to_cobamp_cbm(True)

		red_model, mapping, metabs = self.compress_model(self.cbm, self.singles)

		# map the single reactions from the original network to the new reduced model
		self.exclusion_indices = [[mapping.from_original(k)] for k in
								  [self.cbm.reaction_names.index(n) for n in self.singles] if k in mapping.otn]

		# load a file with validated MCSs in the iAF1260 model
		self.validated = set((k[2:],) for k in open_file('resources/iaf1260/computedmcs.txt', 'r').strip().split('\t\n'))\
						 - {('ATPM',), ('Ec_biomass_iAF1260_core_59p81M',)}

	def sets_are_equal(self, set1, set2):
		# compare the amount of solutions in each set as well as their intersection
		# if all holds true, the enumerated MCSs up to size 1 are correctly identified by this implementation
		return (len(set1) == len(set2) == len(set1 & set2))

	def compress_model(self, cbm, singles):
		# subset reduction algorithm instance
		sr = SubsetReducer()

		# determine blocked reactions using flux variability analysis
		blk_idx = [cbm.reaction_names[i] for i in
				   FluxVariabilityAnalysis(cbm.model).run(0, False, 0).find_blocked_reactions()]

		# create the subset reduction properties instance
		properties = SubsetReducerProperties(keep=singles, block=blk_idx, absolute_bounds=True)

		# create a new reduced model using subset reduction
		return sr.transform(cbm, properties)

	def get_mcs_enumerator_inst(self, red_model, exclusion_indices):
		# create the MCSEnumerator wrapper instance
		return KShortestMCSEnumeratorWrapper(
			model=red_model,
			target_flux_space_dict={  # a dictionary with flux constraints defining the target space
				'Ec_biomass_iAF1260_core_59p81M': (1e-4, None),
				'ATPM': (8.39, 8.39),
				'EX_glc_e_': (-20, None)
			},
			target_yield_space_dict={},  # additional yield constraints (useful in a growth-coupling strategy problem)
			stop_criteria=1,  # iterate 2 times at most
			algorithm_type='kse_populate',  # each iteration yields all solutions of size n, up to stop_criteria
			excluded_solutions=exclusion_indices,  # exclude the single reactions from appearing as MCSs
		)

	def test_size_1_mcs(self):
		red_model, mapping, metabs = self.compress_model(self.cbm, self.singles)
		mcs_enumerator = self.get_mcs_enumerator_inst(red_model, self.exclusion_indices)

		# iterate until stop_criteria and chain the solutions into a single list
		solutions = list(chain(*mcs_enumerator.get_enumerator()))

		# convert reduced model solutions into the original ones
		multiplied_sols = list(chain(*[list(product(*[k.split('_+_') for k in s.keys()])) for s in solutions]))

		# look at the solutions with 1 knockout only
		essentials = set([m for m in multiplied_sols if len(m) == 1])
		self.assertTrue(self.sets_are_equal(essentials, self.validated))


	def test_network_dimensions(self):
		red_model, mapping, metabs = self.compress_model(self.cbm, self.singles)
		self.assertTrue(self.EXPECTED_COMP_SHAPE == red_model.get_stoichiometric_matrix().shape)