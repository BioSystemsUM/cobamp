def identify_conflicts(modelnorm):
	"""
	Identify conflicting constraints in a constraint-based model instantiated with CPLEX
	:param modelnorm: A ConstraintBasedModel instance
	:return: A dict with reaction/metabolite names involved in an irreducible inconsistent subset and their bounds
			(if applicable)
	"""
	try:
		cpxi = modelnorm.model.model.problem
		cpxi.conflict.refine(cpxi.conflict.all_constraints())
		conf = {k[1]: modelnorm.get_reaction_bounds(k[1]) if 'bound' in k[0] else () for k, v in
				dict(zip([(cpxi.conflict.constraint_type[k[1][0][0]], modelnorm.reaction_names[k[1][0][1]]
				if cpxi.conflict.constraint_type[k[1][0][0]] != 'linear' else modelnorm.metabolite_names[k[1][0][1]])
						  for k in cpxi.conflict.get_groups()],
						 [cpxi.conflict.group_status[i] for i in cpxi.conflict.get()])).items()
				if v != 'excluded'}

		# for m in {k: v for k, v in conf.items() if v == ()}.keys():
		# 	matx = modelnorm.get_stoichiometric_matrix(rows=[m])[
		# 		modelnorm.get_stoichiometric_matrix(rows=[m]).nonzero()]
		# 	nzi = modelnorm.get_stoichiometric_matrix(rows=[m]).nonzero()[0]
		# 	invrx = [[c * v for v in vk] for vk, c in
		# 			 zip([modelnorm.get_reaction_bounds(k) for k in nzi], matx.tolist())]
		# 	final_metab_bound = [sum(k) for k in zip(*[(min(x), max(x)) for x in invrx])]
		# 	#print('\t', m, final_metab_bound, '>=', 1)

		#conf_rxs = {i: j for i, j in conf.items() if j != ()}
		return conf
	except:
		print('\tNo conflict')
	finally:
		modelnorm.revert_to_original_bounds()
