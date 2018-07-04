if __name__ == '__main__':

	from metaconvexpy.efm_enumeration.algorithms.kshortest_efms import IrreversibleLinearSystem, KShortestEnumerator
	from metaconvexpy.linear_systems.optimization import DualLinearSystem
	import numpy as np
	import pandas as pd

	S = np.array([[1, -1, 0, 0, -1, 0, -1, 0, 0],
				  [0, 1, -1, 0, 0, 0, 0, 0, 0],
				  [0, 1, 0, 1, -1, 0, 0, 0, 0],
				  [0, 0, 0, 0, 0, 1, -1, 0, 0],
				  [0, 0, 0, 0, 0, 0, 1, -1, 0],
				  [0, 0, 0, 0, 1, 0, 0, 1, -1]])

	irrev = [0, 1, 2, 4, 5, 6, 7, 8]
	T = np.array([0] * S.shape[1]).reshape(1, S.shape[1])
	T[0, 8] = -1
	b = np.array([-1]).reshape(1, )

	dsystem = DualLinearSystem(S, irrev, T, b)
	lsystem = IrreversibleLinearSystem(S, irrev)

	ksh = KShortestEnumerator(lsystem)

	solution_iterator = ksh.population_iterator(8)
	isum_data = []
	data = []
	sol_list = []

	for sols in solution_iterator:
		vals = [sol.var_values() for sol in sols]
		isum = [sol.attribute_value('signed_indicator_sum') for sol in sols]
		isum_data.extend(isum)
		data.extend(vals)
		sol_list.extend(sols)

	#pd.DataFrame(data).to_csv('sols.csv')
	#df = pd.DataFrame(isum_data).apply(lambda x: np.where(x)[0].tolist(), 1)
	df = pd.DataFrame(isum_data)
	df.columns = [0, 1, 2, 4, 5, 6, 7, 8] + [3]
	print(df[sorted(df)])
	ksh.model.write("/home/skapur/MEOCloud/Projectos/MCSEnumeratorPython/problem.lp")