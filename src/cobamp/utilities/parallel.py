from itertools import product
from multiprocessing import cpu_count

from pathos.pools import _ProcessPool

from cobamp.core.optimization import BatchOptimizer

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
    params = {'function': function, 'iterable': sequence}
    if paramargs != None:
        params.update(paramargs)
    return _batch_run(params, threads)

def batch_optimize_cobamp_model(cobamp_model, bounds, objectives, combine_inputs=False, max_threads=MP_THREADS):

    def check_inputs(bounds, objectives):
        all_eq = len(bounds) == len(objectives)
        if combine_inputs or not all_eq:
            bounds, objectives = zip(*product(bounds, objectives))

        obj_coef, senses = zip(*objectives)

        bounds, obj_coef = [{cobamp_model.decode_index(k, 'reaction'): v for k, v in d.items()}
                            for d in (bounds, obj_coef)]

        return bounds, obj_coef, objectives

    cbounds, ccoefs, cobjs = check_inputs(bounds, objectives)
    bopt = BatchOptimizer(cobamp_model.model, threads=min(cbounds, max_threads))
    return bopt.batch_optimize(cbounds, ccoefs, cobjs)
