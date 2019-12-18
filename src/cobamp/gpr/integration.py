from numpy import where, zeros, array

class GeneMatrixBuilder(object):
	def __init__(self, gpr_evaluator):
		self.__gpr_evaluator = gpr_evaluator
		self.__B, self.__b = self.__get_association_matrix()

	def __get_association_matrix(self):
		B = self.__gpr_evaluator.associated_gene_matrix()
		b = B.sum(axis=0)
		return B, b

	def __step_one(self):

		# build step 1 matrices
		# G is for reactions
		# F is for genes

		single_gene_rx = where(self.__b == 1)[0]
		matched_genes = where(self.__B[:, single_gene_rx])[0]

		F1, G1 = [zeros([len(single_gene_rx), n]) for n in self.__B.shape]
		for mat, ind in zip([F1, G1], [matched_genes, single_gene_rx]):
			mat[list(range(len(single_gene_rx))), ind] = 1
		return F1, G1, single_gene_rx

	def __step_exclusive_chars(self, and_ops=True):

		# build step 2 matrices

		exclusive_char_gene_rx = array([i for i in set(where(self.__b > 1)[0]) if
			self.__gpr_evaluator.gpr_has_string(i,self.__gpr_evaluator.or_char()
			if and_ops else self.__gpr_evaluator.and_char())])

		matched_genes, rx_ind = where(self.__B[:, exclusive_char_gene_rx])
		mat_inds = [exclusive_char_gene_rx[i] for i in rx_ind]

		F1, G1 = [zeros([len(exclusive_char_gene_rx), n]) for n in self.__B.shape]
		for mat, ind in zip([F1, G1], [mat_inds, exclusive_char_gene_rx]):
			mat[list(range(len(exclusive_char_gene_rx))), ind] = 1

		return F1, G1, exclusive_char_gene_rx

	def build_gene_reaction_network(self, reactions=None):
		if reactions is None:
			reactions = range(len(self.__gpr_evaluator))
		gpr_lists = [self.__gpr_evaluator.get_gpr_as_lists(i) for i in reactions]


if __name__ == '__main__':
	from cobamp.gpr.evaluator import GPREvaluator, logical_and, logical_or

	gprs = [
		'G1 and G2',
		'G3 or G4',
		'(G1 and G2) or (G3 and G4)',
		'G5',
		'G6'
	]
	# gprs = [model.reactions[i].gene_reaction_rule for i in range(len(model.reactions))]
	gpr_eval = GPREvaluator(gprs, or_fx=logical_or, and_fx=logical_and)
	for i in range(len(gprs)):
		print(gprs[i], gpr_eval.get_gpr_as_lists(i))
