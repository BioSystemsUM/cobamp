def pretty_table_print(table, has_header=True, header_sep=2):
	table = list([list(t) for t in table])

	prntstr = ''
	col_sep_max = ((max([max([len(k) for k in it]) for it in table])//4)+1)
	col_sep = '\t'*col_sep_max

	# check for inconsistent dimensions

	if has_header:
		header = table[0]
		data = table[1:]
		prntstr += col_sep.join(header)+('\n'*header_sep)
	else:
		data = list(table)
	ndata = []
	for line in data:
		tokens = []
		for stri in line:
			separator = '\t'*(col_sep_max - (len(stri) // 4) + 1)
			tokens.extend([stri, separator])
		tokens.pop()
		ndata.append(tokens)

	prntstr += '\n'.join(''.join(k) for k in ndata)
	print(prntstr)


if __name__ == '__main__':
	pretty_table_print([['column1','column2'],['value19999','value2'],['item1item11999','item2'],['item3','']])