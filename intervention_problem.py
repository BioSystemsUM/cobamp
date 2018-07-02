'''
A set of classes to represent constraints applicable to intervention problems
'''

from numpy import zeros, concatenate, array
from itertools import chain
import abc

class InterventionProblem(object):
	def __init__(self, S):
		'''
		Object that generates target matrices for a given set of constraints admissible for an intervention problem
		Args:
			S: The stoichiometric matrix used to generate the enumeration problem
		'''
		self.__num_rx = S.shape[1]

	def generate_target_matrix(self, constraints):
		'''

		Args:
			constraints: An iterable containing valid constraints of

		Returns:

		'''
		constraint_pairs = [const.materialize(self.__num_rx) for const in constraints]
		Tlist, blist = list(zip(*constraint_pairs))

		T = concatenate(Tlist, axis=0)
		b = array(list(chain(*blist)))
		return T,b

class AbstractConstraint(object):
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def materialize(self, n):
		'''

		Args:
			n: Number of columns to include in the target matrix

		Returns: Tuple with Iterable[ndarray] and list of float/int

		'''
		return

class DefaultFluxbound(AbstractConstraint):
	def __init__(self, lb, ub, r_index):
		self.__r_index = r_index
		self.__lb = lb
		self.__ub = ub

	def materialize(self, n):
		Tx = []
		b = []
		if self.__lb != None:
			Tlb = zeros((1,n))
			Tlb[0, self.__r_index] = -1
			b.append(-self.__lb)
			Tx.append(Tlb)
		if self.__ub != None:
			Tub = zeros((1,n))
			Tub[0, self.__r_index] = 1
			b.append(self.__ub)
			Tx.append(Tlb)

		return concatenate(Tx, axis=0), b

class DefaultYieldbound(AbstractConstraint):
	def __init__(self, lb, ub, numerator_index, denominator_index, deviation=0):
		self.__lb = lb
		self.__ub = ub
		self.__numerator_index = numerator_index
		self.__denominator_index = denominator_index
		self.__deviation = deviation

	def materialize(self, n):
		Tx = []
		b = []
		if self.__lb != None:
			Tlb = zeros((1, n))
			Tlb[0, self.__numerator_index] = -1
			Tlb[0, self.__denominator_index] = self.__lb
			b.append(self.__deviation)
			Tx.append(Tlb)
		if self.__ub != None:
			Tub = zeros((1, n))
			Tub[0, self.__numerator_index] = 1
			Tub[0, self.__denominator_index] = - self.__ub
			b.append(self.__deviation)
			Tx.append(Tub)

		return concatenate(Tx,axis=0), b


if __name__ == '__main__':
	ip = InterventionProblem(zeros([20, 20]))
	constraints = [DefaultFluxbound(0, 10, 14), DefaultFluxbound(0, None, 16), DefaultYieldbound(None, 2, 3, 5)]
	T, b = ip.generate_target_matrix(constraints)

