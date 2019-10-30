if __name__ == '__main__':
	from cobamp.utilities.file_utils import read_pickle
	from cobamp.core.optimization import BatchOptimizer
	from numpy import array

	mobjr = read_pickle('resources/models/Recon2_v04_pruned.xml.objrdr')
	S = mobjr.get_stoichiometric_matrix()
	lb, ub = map(array, mobjr.get_model_bounds(separate_list=True))

	model = mobjr.to_cobamp_cbm('CPLEX')
	obj_rx = model.reaction_names.index('biomass_reaction')

	n_iterations = len(model.reaction_names)
	bounds = [{k:(0,0)} for k in range(len(model.reaction_names))]
	objective_coefs = [{obj_rx: 1} for _ in range(n_iterations)]
	objective_sense = [False for _ in range(n_iterations)]

	batch_opt  = BatchOptimizer(model.model, threads=12)
	res = batch_opt.batch_optimize(bounds, objective_coefs, objective_sense)
	print(res)

