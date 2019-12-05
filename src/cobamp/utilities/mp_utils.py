from pathos.pools import _ProcessPool
from multiprocessing import cpu_count

MP_THREADS = cpu_count()

def _batch_function(param_index):
	global _params, _function, _iterable
	return param_index, _function(_iterable[param_index], _params)


def _pool_initializer(params):
	global _iterable, _params, _function
	_params = params
	_iterable = params['iterable']
	_function = params['function']

def _batch_run(params, threads):
	jobs = len(params['iterable'])
	res_map = [None for _ in range(jobs)]
	true_threads = min((jobs // 2) + 1, threads)
	it_per_job = jobs // threads
	pool = _ProcessPool(
		processes=true_threads,
		initializer=_pool_initializer,
		initargs=([params])
	)
	for i, value in pool.imap_unordered(_batch_function, list(range(jobs)),
										chunksize=it_per_job):
		res_map[i] = value

	pool.close()
	pool.join()

	return res_map

def batch_run(function, sequence, paramargs=None, threads=MP_THREADS):
	params = {'function':function, 'iterable':sequence}
	if paramargs != None:
		params.update(paramargs)
	return _batch_run(params, threads)