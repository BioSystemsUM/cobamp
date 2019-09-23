from itertools import chain
from numpy import zeros, where, array

def logical_or(x):
	return sum(x) >= 1

def logical_and(x):
	return sum(x) == len(x)


def aux_apply(fx, it):
	print(fx,it)
	args = [k for k in it if k is not None]
	return fx(args) if args else None

def convert_gpr_to_list(gpr, apply_fx=str, or_char = 'or', and_char = 'and'):
	proteins = list(
		map(
			lambda x: x.strip().replace('(', '').replace(')', ''),
			apply_fx(gpr).split(or_char)
		)
	)

	rules = [[s.strip() for s in x.split(and_char) if s.strip() != ''] for x in proteins]
	return rules


class GPREvaluator(object):
	def __init__(self, gpr_list, or_fx=logical_or, and_fx=logical_and, apply_fx=str, or_char='or', and_char='and'):
		self.__gprs = gpr_list
		self.or_fx, self.and_fx = or_fx, and_fx
		self.__or_char, self.__and_char = or_char, and_char
		self.apply_fx = apply_fx
		self.__gpr_list = [convert_gpr_to_list(g, apply_fx=self.apply_fx, or_char=or_char, and_char=and_char) for g in self.__gprs]
		self.__genes = 	tuple(set(list(chain(*chain(*self.__gpr_list)))))
		self.__gene_to_index_mapping = dict(zip(self.__genes,range(len(self.__genes))))

	def get_num_of_gprs(self):
		return len(self.__gpr_list)

	def gpr_has_string(self, index, string):
		return string in self.__gprs[index]

	def get_gpr_as_lists(self, index):
		return self.__gpr_list[index]

	def or_char(self):
		return self.__or_char

	def and_char(self):
		return self.__and_char

	def get_genes(self):
		return self.__genes

	def eval_gpr(self, index, state):
		print(self.__gpr_list[index])
		return aux_apply(self.or_fx,
				[aux_apply(self.and_fx, [state[x] for x in gs if x in state.keys()]) for gs in self.__gpr_list[index]])

	def associated_genes(self, index):
		return list(chain(*self.__gpr_list[index]))

	def associated_gene_matrix(self):
		B = zeros([len(self.__genes), len(self.__gpr_list)])
		row_ind, col_ind = [], []
		for i in range(self.get_num_of_gprs()):
			gene_indexes = [self.__gene_to_index_mapping[k] for k in self.associated_genes(i)]
			row_ind += gene_indexes
			col_ind += [i]*len(gene_indexes)
		B[row_ind, col_ind] = 1
		return B

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
		for mat, ind in zip([F1,G1],[matched_genes, single_gene_rx]):
			mat[list(range(len(single_gene_rx))), ind] = 1
		return F1, G1, single_gene_rx

	def __step_exclusive_chars(self, and_ops=True):

		# build step 2 matrices

		exclusive_char_gene_rx = array([i for i in set(where(self.__b > 1)[0]) if self.__gpr_evaluator.gpr_has_string(i, self.__gpr_evaluator.or_char() if and_ops else self.__gpr_evaluator.and_char())])
		matched_genes, rx_ind = where(self.__B[:, exclusive_char_gene_rx])
		mat_inds = [exclusive_char_gene_rx[i] for i in rx_ind]

		F1, G1 = [zeros([len(exclusive_char_gene_rx), n]) for n in self.__B.shape]
		for mat, ind in zip([F1,G1],[mat_inds, exclusive_char_gene_rx]):
			mat[list(range(len(exclusive_char_gene_rx))), ind] = 1

		return F1, G1, exclusive_char_gene_rx

	def build_gene_reaction_network(self, reactions):
		gpr_lists = [self.__gpr_evaluator.get_gpr_as_lists(i) for i in reactions]


# def __identify_single_gene_reactions(self):
	#
	# 	for i in self.__gpr_evaluator.get_num_of_gprs()
	#

if __name__ == '__main__':
	gprs = [
		'G1 and G2',
		'G3 or G4',
		'(G1 and G2) or (G3 and G4)',
		'G5',
		'G6'
	]
	#gprs = [model.reactions[i].gene_reaction_rule for i in range(len(model.reactions))]
	gpr_eval = GPREvaluator(gprs, or_fx=logical_or, and_fx=logical_and)
	for i in range(len(gprs)):
		print(gprs[i], gpr_eval.get_gpr_as_lists(i))