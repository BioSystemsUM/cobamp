from typing import Iterable

def get_dict_hash(x: dict, hash_function=hash):
    return hash_function(frozenset(x.items()))

def get_unique_dicts(dict_iterable: Iterable[dict], hash_function=hash):
    orig_hashes = []
    cache = {}

    for d in dict_iterable:
        dhash = get_dict_hash(d, hash_function)
        if dhash not in cache:
            cache[dhash] = d
        orig_hashes.append(dhash)
    return cache, orig_hashes