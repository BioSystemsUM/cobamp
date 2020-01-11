from numpy import where, zeros, array, vstack, unique, eye, hstack, ones
from cobamp.wrappers.method_wrappers import KShortestGenericMCSEnumeratorWrapper
from cobamp.core.models import ConstraintBasedModel
from functools import reduce
def filled_vector(dim, index, fill_value=1):
	fvec = zeros(dim)
	fvec[index] = fill_value
	return fvec


class GeneMatrixBuilder(object):
	def __init__(self, gpr_evaluator):
		self.__gpr_evaluator = gpr_evaluator
		self.__B, self.__b = self.get_association_matrix()

	def get_association_matrix(self):
		B = self.__gpr_evaluator.associated_gene_matrix()
		b = B.sum(axis=0)
		return B, b

	@staticmethod
	def gpr_single_gene(gpr_list):
		return len(gpr_list) == 1 and len(gpr_list[0]) == 1

	@staticmethod
	def gpr_only_and_rules(gpr_list):
		return len(gpr_list) == 1 and len(gpr_list[0]) > 1

	@staticmethod
	def gpr_only_or_rules(gpr_list):
		return len(gpr_list) > 1 and not (False in ([len(g) <= 1 for g in gpr_list]))

	def optimize_dual_gpr_graph(self, F):
		# S -> H -> subset of G
		cbm = self.get_gpr_model(F)
		dual_mat = zeros([F.shape[1],len(cbm.reaction_names)])
		dual_mat[:F.shape[1],:F.shape[1]] = eye(F.shape[1])
		wrp = KShortestGenericMCSEnumeratorWrapper(
			model=cbm, target_flux_space_dict={'GPRs':(1,None)}, target_yield_space_dict={},
			dual_matrix=dual_mat, dual_var_mapper={i:i for i in range(F.shape[1])}, stop_criteria=F.shape[1],
			algorithm_type='kse_populate'
		)
		enum = wrp.get_enumerator()
		sols = list(map(lambda d: array(list(d.keys())),reduce(lambda x,y: x+y, enum)))
		return sols

	@staticmethod
	def get_gpr_model(F):
		c,g = F.shape

		S = vstack([hstack(r) for r in [
			[eye(g), -F.T, zeros([g,c+1])],
			[zeros([c, g]), eye(c), -eye(c), zeros([c, 1])],
			[zeros([1, c+g]), ones([1,c]), -ones([1,1])]
		]])

		bounds = [[0, None] for _ in range(S.shape[1])]
		mn = ['MG'+str(i) for i in range(g)] + ['MC'+str(i) for i in range(c)] + ['GPR']
		rn = ['SG'+str(i) for i in range(g)] + ['SC'+str(i) for i in range(c)] + ['OC'+str(i) for i in range(c)] + \
			 ['GPRs']
		return ConstraintBasedModel(S, bounds, reaction_names=rn, metabolite_names=mn)

	def get_GF_matrices(self):
		gpr_eval = self.__gpr_evaluator
		genes = dict(zip(gpr_eval.get_genes(), range(len(gpr_eval.get_genes()))))
		total_g, total_f = [], []
		for i in range(len(gpr_eval) - 1):
			gpr_as_list = gpr_eval.get_gpr_as_lists(i)
			# AND means new row
			# OR means all active indices in the same row
			if not (len(gpr_as_list) == 0 or (len(gpr_as_list) == 1 and len(gpr_as_list[0]) == 0)):
				alt_case = [f(gpr_as_list) for f in
							[self.gpr_single_gene, self.gpr_only_and_rules, self.gpr_only_or_rules]]
				cur_g, cur_f = [], []
				if not (True in alt_case) and len(gpr_as_list) > 0:
					for complex in gpr_as_list:
						if len(complex) > 0:
							flist = filled_vector(len(genes), [genes[g] for g in complex], fill_value=1)
							glist = filled_vector(len(gpr_eval), [i], fill_value=1)
							cur_g.append(glist)
							cur_f.append(flist)

					F_sub_full = vstack(cur_f)
					F_genes = unique(F_sub_full.nonzero()[1])
					F_temp = F_sub_full[:, F_genes]
					row_inds = [F_genes[k] for k in self.optimize_dual_gpr_graph(F_temp)]
					cur_f = [filled_vector(len(genes), inds, fill_value=1) for inds in row_inds]
					cur_g = [filled_vector(len(gpr_eval), [i], fill_value=1) for _ in row_inds]
				else:
					if len(gpr_as_list) > 0  and len(gpr_as_list[0]) > 0:
						if alt_case[2]:
							flist = [filled_vector(len(genes), [genes[c[0]] for c in gpr_as_list], fill_value=1)]
							glist = [filled_vector(len(gpr_eval), [i], fill_value=1)]
						elif alt_case[1]:
							flist = [filled_vector(len(genes), [genes[g]], fill_value=1) for g in gpr_as_list[0]]
							glist = [filled_vector(len(gpr_eval), [i], fill_value=1)]*len(gpr_as_list[0])
						elif alt_case[0]:
							flist = [filled_vector(len(genes), [genes[gpr_as_list[0][0]]], fill_value=1)]
							glist = [filled_vector(len(gpr_eval), [i], fill_value=1)]
						else:
							flist, glist = [], []
						cur_g.extend(glist)
						cur_f.extend(flist)
				total_f.extend(cur_f)
				total_g.extend(cur_g)

		G, F = [vstack(l).astype(bool) for l in [total_g, total_f]]

		Fm, old_indices, reverse = unique(F, axis=0, return_index=True, return_inverse=True)
		Frev_dict = {k: where(reverse == k)[0].tolist() for k in unique(reverse)}
		Gm = vstack([G[idx, :].any(axis=0) for k, idx in Frev_dict.items()])

		return [x.astype(int) for x in [Gm, Fm]] + [genes]



if __name__ == '__main__':
	from cobamp.gpr.evaluator import GPREvaluator, logical_and, logical_or

	gprs = [
		'g1',
		'g2',
		'g2',
		'g3 and g4',
		'g2 and g5',
		'g3 or g6',
		'(g2 and (g5 or g6)) or g7',
		''
	]
	gpr_eval = GPREvaluator(gprs)
	gmb = GeneMatrixBuilder(gpr_eval)
	G, F, gene_mapping = gmb.get_GF_matrices()
	ord_map = [gene_mapping['g'+str(i)] for i in range(1,8)]
	G[:,ord_map]
	F[:,ord_map]
	# revmap = {v:k for k,v in gene_mapping.items()}
	# [[revmap[k] for k in r.nonzero()[0]] for r in F]



