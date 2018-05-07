'''
Module containing nullspace analysis tools.

Compression pipeline:
    - remove conservation relations
    - remove FVA blocked reactions
    - generate kernel
    - remove kernel blocked reactions
    - generate subset matrix
    - populate subsets

'''
from numpy import abs, where, compress, concatenate, ones, array, dot
from numpy.linalg import svd
from sympy import Matrix

EPSILON = 1e-10
PRECISION = 1e-10

def compute_nullspace(A, eps=1e-9, left=True):
    u, s, v = svd(A)
    padding = max(0, A.shape[1] - s.shape[0])
    mask = concatenate(((s <= eps), ones((padding,), dtype=bool)), axis=0)
    return compress(mask, u.T, 0) if left else compress(mask, v.T, 1)

def nullspace_blocked_reactions(K, tolerance):
    return where(sum(abs(K.T) > tolerance) == K.shape[0])[0]

if __name__ == '__main__':
    A = array([[2, 3, 5], [-4, 2, 3], [0, 0, 0]])
    nullspace = compute_nullspace(A, left=False)
