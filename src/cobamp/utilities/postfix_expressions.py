from operator import add, sub, mul, truediv, pow


class Stack(list):

	def push(self, x):
		self.append(x)

	def top(self):
		return self[-1]


class Queue(list):

	def push(self, x):
		self.insert(0, x)

	def top(self):
		return self[-1]


def parse_infix_expression(op, is_operand_fx, is_operator_fx, precedence_fx):
	tokens = Queue(op[::-1])
	output = Queue()
	opstk = Stack()
	while tokens:
		token = tokens.pop()
		if is_operand_fx(token):
			output.push(token)
		elif is_operator_fx(token):
			while opstk and ((opstk.top() != '(') and ((precedence_fx(token) < precedence_fx(opstk.top())) or (
					left_operator_association(opstk.top()) and (precedence_fx(token) == precedence_fx(opstk.top()))))):
				output.push(opstk.pop())
			opstk.push(token)
		elif token == '(':
			opstk.push(token)
		elif token == ')':
			while opstk and (opstk.top() != '('):
				output.push(opstk.pop())
			if opstk.top() != '(':
				print('Mismatched parentheses found!')
			else:
				opstk.pop()
	while opstk:
		op_remaining = opstk.pop()
		if op_remaining in ('(', ')'):
			print('Mismatched parentheses found')
		output.push(op_remaining)
	return list(output)[::-1]


def evaluate_postfix_expression(op, eval_fx, type_conv=int):
	stk = Stack()
	for token in op:
		if is_operator_token(token):
			o1, o2 = type_conv(stk.pop()), type_conv(stk.pop())
			result = eval_fx(token, o1, o2)
			stk.push(result)
		elif is_number_token(token):
			stk.push(token)
	return stk.pop()


def tokenize_infix_expression(inf_exp_str):
	return list(filter(lambda x: x != '', inf_exp_str.replace('(', ' ( ').replace(')', ' ) ').split(' ')))


def tokenize_boolean_expression(inf_exp_str, default_value='1'):
	return [tok if tok in ('0', '1', 'and', 'not', 'or', ')', '(') else default_value for tok in
			tokenize_infix_expression(inf_exp_str)]


def is_number_token(token):
	return token.replace('.', '', 1).replace('-', '', 1).isnumeric()


def is_operator_token(token):
	return token in ['**', '/', '*', '+', '-']


def op_prec(op):
	precedence = {
		'**': 4,
		'/': 3,
		'*': 3,
		'+': 2,
		'-': 1}
	return precedence[op]


def left_operator_association(op):
	return False if op == '**' else True


operators = {
	'+': add,
	'-': sub,
	'*': mul,
	'/': truediv,
	'**': lambda a, b: pow(b, a)
}


def eval_math_operator(operator, o1, o2):
	return operators[operator](o1, o2)


def is_boolean_value(token):
	if token in ('1', '0'):
		return True
	elif token.isnumeric():
		print('Illegal token', token, 'found.')
		return False
	else:
		return False


def is_boolean_operator(token):
	return token.upper() in ['AND', 'OR', 'NOT']


def boolean_precedence(token):
	pdict = {'not': 3,
			 'and': 2,
			 'or': 1}
	return pdict[token]


truth_table = {
	'and': {
		(0, 0): 0,
		(1, 0): 0,
		(0, 1): 0,
		(1, 1): 1
	},
	'or': {
		(0, 0): 0,
		(1, 0): 1,
		(0, 1): 1,
		(1, 1): 1
	},
	'not': {
		0: 1,
		1: 0
	}
}


def eval_boolean_operator(operator, o1, o2):
	return truth_table[operator][(o1, o2)]


if __name__ == '__main__':

	def test_algebraic_expression():
		op = tokenize_infix_expression('((15 / (7 - (1 + 1))) * 3) - (2 + (1 + 1)) ')
		psfix = parse_infix_expression(op, is_number_token, is_operator_token)
		res = evaluate_postfix_expression(psfix, eval_math_operator)
		print(res)


	def test_boolean_expression():
		from urllib.request import urlretrieve
		from cobra.io.sbml3 import read_sbml_model
		from random import random

		path, content = urlretrieve('http://bigg.ucsd.edu/static/models/RECON1.xml')
		model = read_sbml_model(path)
		ogpr = model.reactions.ATPS4m.gene_name_reaction_rule
		gene_activation = {k: 1 for k in [g.id for g in model.genes]}
		for test in range(20):
			gpr = ogpr
			for gene in gene_activation:
				dec = random() > 0.6
				gpr = gpr.replace(gene, '1' if dec else '0')
			op = [tok if tok in ('0', '1', 'and', 'not', 'or', ')', '(') else '1' for tok in
				  tokenize_infix_expression(gpr)]
			psfix = parse_infix_expression(op, is_boolean_value, is_boolean_operator, boolean_precedence)
			print(gpr, ''.join(psfix), evaluate_postfix_expression(psfix, eval_boolean_operator), sep=',')
