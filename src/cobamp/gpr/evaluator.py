import warnings

from itertools import chain
from numpy import zeros, where, array
from boolean.boolean import BooleanAlgebra
import re

GPR_GENE_RE = re.compile("\\b(?!and\\b|or\\b|AND\\b|OR\\b)[([A-z0-9\.]*")
BOOLEAN_ALGEBRA = BooleanAlgebra()


def logical_or(x):
	return sum(x) >= 1


def logical_and(x):
	return sum(x) == len(x)


def aux_apply(fx, it):
	#print(fx, it)
	args = [k for k in it if k is not None]
	return fx(args) if args else None

def normalize_boolean_expression(rule):
	try:
		expression = BOOLEAN_ALGEBRA.parse(rule, simplify=True)
		bool_expression = BOOLEAN_ALGEBRA.normalize(expression, BOOLEAN_ALGEBRA.OR)
		return str(bool_expression).replace('&', ' and ').replace('|', ' or ')
	except Exception as e:
		warnings.warn('Could not normalize this rule: ' + rule)
		return rule

def convert_gpr_to_list(gpr, apply_fx=str, or_char='or', and_char='and'):
	proteins = list(
		map(
			lambda x: x.strip().replace('(', '').replace(')', ''),
			apply_fx(gpr).split(or_char)
		)
	)
	rules = [[s.strip() for s in x.split(and_char) if s.strip() != ''] for x in proteins]
	return rules


class GPREvaluator(object):
	def __init__(self, gpr_list, apply_fx=str, or_char='or', and_char='and', ttg_ratio=20):
		self.apply_fx = apply_fx
		self.__or_char, self.__and_char = or_char, and_char
		self.ttg_ratio = ttg_ratio



	def __initialize(self, gpr_list):
		self.__gprs = []
		self.__gpr_list = []
		self.add_gprs(gpr_list)
		self.__update_genes()

	def add_gprs(self, gpr_list):
		gprs = [self.__preprocess_gprs(gp, token_to_gene_ratio=self.ttg_ratio) if gp != '' else '' for gp in gpr_list]
		gpr_list = [convert_gpr_to_list(g, apply_fx=str, or_char=self.__or_char, and_char=self.__and_char) for g in
		 self.__gprs]

		self.__gprs.extend(gprs)
		self.__gpr_list.extend(gpr_list)
		self.__update_genes()

	def remove_gprs(self, indices):
		self.__gprs, self.__gpr_list = zip(*[(self.__gprs[i], self.__gpr_list[i]) for i in range(len(self.__gprs)) if i not in indices])
		self.__gprs, self.__gpr_list = [list(k) for k in [self.__gprs, self.__gpr_list]]
		self.__update_genes()

	def __update_genes(self):
		self.__genes = tuple(set(list(chain(*chain(*self.__gpr_list)))))
		self.__gene_to_index_mapping = dict(zip(self.__genes, range(len(self.__genes))))

	def __preprocess_gprs(self, gpr_str, token_to_gene_ratio=20):
		def fix_name(gpr_string):
			matches = [k for k in GPR_GENE_RE.finditer(gpr_string) if k.span()[0] - k.span()[1] != 0]
			unique_tokens = set([m.string for m in matches])
			for offset, match_obj in enumerate(matches):
				final_pos = match_obj.span()[0] + offset
				gpr_string = gpr_string[:final_pos] + '_' + gpr_string[final_pos:]
			return gpr_string, len(matches), len(unique_tokens), unique_tokens

		# gpr_list = []
		# for gpr_str in gpr_string_list:
		rule, num_matches, num_unique_tokens, unique_tokens = fix_name(gpr_str)
		if self.apply_fx:
			rule = self.apply_fx(rule)
		if (num_unique_tokens > 0) and (num_matches // num_unique_tokens) < token_to_gene_ratio:
			rule = normalize_boolean_expression(rule)
		else:
			warnings.warn(
				'Will not normalize rules with more than ' + str(token_to_gene_ratio) + ' average tokens per gene')

		matches_post = [k for k in GPR_GENE_RE.finditer(rule) if
						(k.span()[0] - k.span()[1] != 0) and k.string[k.span()[0]:k.span()[1]][0] == '_']
		for offsetp, matchp_obj in enumerate(matches_post):
			final_pos = matchp_obj.span()[0] - offsetp
			rule = rule[:final_pos] + rule[final_pos + 1:]

		return rule


	def __len__(self):
		return len(self.__gpr_list)

	def __getitem__(self, item):
		return self.__gprs[item]

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

	def eval_gpr(self, index, state, or_fx=logical_or, and_fx=logical_and):
		#print(self.__gpr_list[index])
		return aux_apply(or_fx,
						 [aux_apply(and_fx, [state[x] for x in gs if x in state.keys()]) for gs in
						  self.__gpr_list[index]])

	def associated_genes(self, index):
		return list(chain(*self.__gpr_list[index]))

	def associated_gene_matrix(self):
		B = zeros([len(self.__genes), len(self.__gpr_list)])
		row_ind, col_ind = [], []
		for i in range(len(self)):
			gene_indexes = [self.__gene_to_index_mapping[k] for k in self.associated_genes(i)]
			row_ind += gene_indexes
			col_ind += [i] * len(gene_indexes)
		B[row_ind, col_ind] = 1
		return B



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
	# gprs = [model.reactions[i].gene_reaction_rule for i in range(len(model.reactions))]
	gpr_eval = GPREvaluator(gprs, or_fx=logical_or, and_fx=logical_and)
	for i in range(len(gprs)):
		print(gprs[i], gpr_eval.get_gpr_as_lists(i))
