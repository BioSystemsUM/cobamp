if __name__ == '__main__':

	from pathway_analysis import IrreversibleLinearSystem, DualLinearSystem, KShortestEnumerator
	import numpy as np
	import pandas as pd
	import pprofile

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

	profiler = pprofile.Profile()
	with profiler:
		ksh = KShortestEnumerator(dsystem)

	solution_iterator = ksh.population_iterator(3)
	data = []
	sol_list = []
	for sols in solution_iterator:
		isums = [sol.attribute_value('indicator_sum') for sol in sols]
		data.extend(isums)
		sol_list.extend(sols)

	df = pd.DataFrame(data).apply(lambda x: np.where(x)[0].tolist(), 1)
	print(df)
