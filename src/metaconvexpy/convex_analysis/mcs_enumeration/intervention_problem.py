'''
A set of classes to represent constraints applicable to intervention problems
'''

from numpy import zeros, concatenate, array
from itertools import chain
import abc

class InterventionProblem(object):
	'''
	Class containing functions useful when defining an problem using the intervention problem framework.
	References:
		[1] Hädicke, O., & Klamt, S. (2011). Computing complex metabolic intervention strategies using constrained
		minimal cut sets. Metabolic engineering, 13(2), 204-213.
	'''
	def __init__(self, S):
		'''
		Object that generates target matrices for a given set of constraints admissible for an intervention problem
		Parameters:
			S: The stoichiometric matrix used to generate the enumeration problem
		'''
		self.__num_rx = S.shape[1]

	def generate_target_matrix(self, constraints):
		'''

		Parameters:
			constraints: An iterable containing valid constraints of

		Returns:

		'''
		constraint_pairs = [const.materialize(self.__num_rx) for const in constraints]
		Tlist, blist = list(zip(*constraint_pairs))

		T = concatenate(Tlist, axis=0)
		b = array(list(chain(*blist)))
		return T,b

class AbstractConstraint(object):
	'''
	Object representing a constraint to be used within the intervention problem structures provided in this package.
	'''
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def materialize(self, n):
		'''
		Generates a matrix T 1-by-n or 2-by-n and a list b of length 1 or 2 to be used for target flux vector
		definition within the intervention problem framework
		Parameters:
			n: Number of columns to include in the target matrix

		Returns: Tuple with Iterable[ndarray] and list of float/int

		'''
		return

	@abc.abstractmethod
	def from_tuple(tup):
		"""
		Generates a constraint from a tuple. Refer to subclasses for each specific format.

		Returns
		-------

		"""
		return

class DefaultFluxbound(AbstractConstraint):
	'''
	Class representing bounds for a single flux with a lower and an upper bound.
	'''
	def __init__(self, lb, ub, r_index):
		'''
		Parameters
		----------
		lb: Numerical lower bound
		ub: Numerical upper bound
		r_index: Reaction index on the stoichiometric matrix to which this bound belongs
		'''

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
			Tx.append(Tub)

		return concatenate(Tx, axis=0), b

	def from_tuple(tup):
		"""

		Returns a DefaultFluxbound instance from a tuple containing a reaction index as well as lower and upper bounds.
		-------

		"""
		index, lb, ub = tup
		return DefaultFluxbound(lb, ub, index)

class DefaultYieldbound(AbstractConstraint):
	'''
	Class representing a constraint on a yield between two fluxes. Formally, this constraint can be represented as
	n - yd < b, assuming n and d as the numerator and denominator fluxes (yield(N,D) = N/D), y as the maximum yield and
	b as a deviation value.
	'''
	def __init__(self, lb, ub, numerator_index, denominator_index, deviation=0):
		'''

		Parameters
		----------
		lb: numerical lower bound
		ub: numerical upper bound
		numerator_index: reaction index for the flux in the numerator
		denominator_index: reaction index for the flux in the denominator
		deviation: numerical deviation for the target space
		'''
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

	def from_tuple(tup):
		"""

		Returns a DefaultYieldbound instance from a tuple containing numerator and denominator indices, yield lower and
		upper bounds, a flag indicating whether it's an upper bound and a deviation (optional)
		-------

		"""
		n, d, ylb, yub = tup[:4]
		if len(tup) > 4:
			dev = tup[4]

		return DefaultYieldbound(ylb, yub, n, d, dev)


if __name__ == '__main__':
	ip = InterventionProblem(zeros([20, 20]))
	constraints = [DefaultFluxbound(0, 10, 14), DefaultFluxbound(0, None, 16), DefaultYieldbound(None, 2, 3, 5)]
	T, b = ip.generate_target_matrix(constraints)

