'''
	Implementation of the fast flux variability analysis method by Gudmundsson and Thiele (Computationally efficient
	flux variability analysis - BMC Bioinformatics 2010 11:489).

	Some parts have been heavily inspired by the talented people supporting cobrapy.

'''

from cobamp.core.optimization import LinearSystemOptimizer
from numpy import zeros
from pathos.pools import _ProcessPool
from pathos.multiprocessing import cpu_count

def _fva_initializer(linear_system, sense, gamma):
	global _linear_system
	global _opt
	global _sense
	global _lenrx
	global _gamma
	_linear_system = linear_system
	_lenrx = _linear_system.get_stoich_matrix_shape()[1]
	_opt = LinearSystemOptimizer(_linear_system, build=False)
	_sense = sense
	_gamma = gamma


def _fva_iteration(i):
	global _linear_system
	global _opt
	global _sense
	global _gamma
	global _lenrx
	#print('Iterating for ',i)
	w = zeros( _lenrx).ravel()
	w[i] = 1
	_linear_system.set_objective(w, _sense)
	sol = _opt.optimize()
	#_linear_system.add_rows_to_model(w, [sol.objective_value()* _gamma], [None], only_nonzero=True)
	# print(w, _sense, i, sol.x()[i], sol.objective_value())
	return i, sol.objective_value()



class FluxVariabilityAnalysis(object):
	def __init__(self, linear_system, workers=None):
		self.ls = linear_system
		self.n_jobs = min(cpu_count() if workers ==	None else workers, linear_system.get_stoich_matrix_shape()[1])

	def run(self, initial_objective, minimize_initial, gamma=1-1e-6):

		M,N = self.ls.get_stoich_matrix_shape()
		result = {i:[0,0] for i in range(N)}
		opt = LinearSystemOptimizer(self.ls, build=False)
		c = zeros(N)
		c[initial_objective] = 1
		self.ls.set_objective(c, minimize_initial)

		v0 = opt.optimize()
		z0 = v0.objective_value()

		self.ls.add_rows_to_model(c.reshape([1, N]), [z0 * gamma], [None], only_nonzero=True,
										 names=['FASTFVAINITIALCONSTRAINT'])

		for sense in [True, False]:
			rx_per_job = N // self.n_jobs
			self.pool = _ProcessPool(
				processes=self.n_jobs,
				initializer=_fva_initializer,
				initargs=(self.ls, sense, gamma)
			)
			for i, value in self.pool.imap_unordered(_fva_iteration, range(N), chunksize=rx_per_job):
				result[i][int(not sense)] = value

			self.pool.close()
			self.pool.join()

		self.ls.remove_from_model([M], 'const')
		return [result[i] for i in range(N)]


if __name__ == '__main__':
	from cobra.io.sbml3 import read_sbml_model
	from cobamp.wrappers import COBRAModelObjectReader
	import time

	model =  read_sbml_model('/home/skapur/MEOCloud/Projectos/cobamp/examples/iAF1260_resources/original_model/Ec_iAF1260_flux2.xml')
	mor = COBRAModelObjectReader(model)

	cbm_mp = mor.to_cobamp_cbm('CPLEX')
	cbm_fast = mor.to_cobamp_cbm('CPLEX')

	init_sol = cbm_mp.optimize({1004:1}, False)
	Z0 = (1-1e-6) * init_sol.objective_value()
	cbm_mp.set_reaction_bounds(1004, lb=Z0)

	c1_time = time.time()

	pp = _ProcessPool(cpu_count())

	limits_mp = list(pp.map(cbm_mp.flux_limits, range(len(cbm_mp.reaction_names))))
	pp.close()
	pp.join()
	c2_time = time.time()
	print('Multi-threaded:',c2_time-c1_time,'seconds')

	fva = FluxVariabilityAnalysis(cbm_fast.model)
	limits_fast = fva.run(1004, False)
	c3_time = time.time()
	print('Multi-threaded fast FVA:',c3_time-c2_time,'seconds')

	error = 1e-6
	error_rx = []
	for i, lsts in enumerate(zip(limits_mp, limits_fast)):
		mpr, fr = lsts
		ld, ud = [mpr[i] - fr[i] for i in range(len(mpr))]
		if (abs(ld) > error) | (abs(ud) > error):
			error_rx.append([i, mpr, fr])

	print('Valid:',len(error_rx) == 0)