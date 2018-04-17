from sympy import Matrix
from numpy import array, float64

def compute_nullspace(A):
    return array(Matrix(A).nullspace()[0]).astype(float64)

if __name__ == '__main__':
    A = [[2, 3, 5], [-4, 2, 3], [0, 0, 0]]
    nullspace = compute_nullspace(A)
    assert(sum(array([[-0.0625],
               [-1.625],
               [1.]]) == compute_nullspace(A))[0] == 3)