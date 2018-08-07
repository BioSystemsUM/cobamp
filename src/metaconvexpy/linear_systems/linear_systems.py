from itertools import chain
from metaconvexpy.utilities.set_utils import is_subset, has_no_overlap
import cplex
import numpy as np

from metaconvexpy.linear_systems.optimization import CPLEX_INFINITY

class IrreversibleLinearSystem(object):
	def __init__(self, S, irrev, extracellular_metabs=None, medium_metabs=None, consumed_metabs=None, produced_metabs=None):
		self.__model = cplex.Cplex()

		self.__extracellular_metabs = set() if extracellular_metabs is None else set(extracellular_metabs)
		self.__medium_metabs = set() if medium_metabs is None else set(medium_metabs)
		self.__consumed_metabs = set() if consumed_metabs is None else set(consumed_metabs)
		self.__produced_metabs = set() if produced_metabs is None else set(produced_metabs)


		if extracellular_metabs is not None:
			# assert is_subset(consumed_metabs, medium_metabs)
			# assert is_subset(produced_metabs, medium_metabs)
			# assert is_subset(medium_metabs, extracellular_metabs)
			# assert has_no_overlap(consumed_metabs, produced_metabs)

			self.__ec_to_remove = set(extracellular_metabs) - medium_metabs - consumed_metabs - produced_metabs



		self.__ivars = None
		self.S, self.irrev = S, irrev
		self.__c = "C"

	def get_dvar_mapping(self):
			return self.__dvar_mapping

	def get_model(self):
		return self.__model

	def get_stoich_matrix_shape(self):
		return self.S.shape

	def get_dvars(self):
		return self.__dvars

	def get_c_variable(self):
		return self.__c

	def build_problem(self):
		# Defining useful length constants
		nM, nR = self.S.shape
		nIrrev = len(self.irrev)
		nRev = nR - nIrrev
		veclens = [("irr", nIrrev), ("revfw", nRev), ("revbw", nRev)]
		Sxi, Sxr = self.S[:, self.irrev], np.delete(self.S, self.irrev, axis=1)
		S_full = np.concatenate([Sxi, Sxr, -Sxr], axis=1)

		vi, vrf, vrb = [[(pref + str(i), 0, CPLEX_INFINITY) for i in range(n)] for pref, n in veclens]

		vd = chain(vi, vrf, vrb)
		names, lb, ub = list(zip(*vd))
		self.__model.variables.add(names=names, lb=lb, ub=ub)

		np_names = np.array(names)
		nnz = list(map(lambda y: np.nonzero(y)[1], zip(S_full)))

		lin_expr = [cplex.SparsePair(ind=np_names[x].tolist(), val=row[x].tolist()) for row, x in zip(S_full, nnz)]

		rhs = [0] * S_full.shape[0]
		senses = 'E'* S_full.shape[0]
		cnames = ['C_' + str(i) for i in range(S_full.shape[0])]

		self.__model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)
		self.__model.variables.add(names=['C'], lb=[1], ub=[CPLEX_INFINITY])
		self.__dvars = vi + list(map(list, zip(vrf, vrb)))

		vi_names = list(zip(*vi))[0]
		vrf_names = list(zip(*vrf))[0]
		vrb_names = list(zip(*vrb))[0]

		self.__dvars = list(vi_names) + list(zip(vrf_names,vrb_names))
		var_index_sequence = (self.irrev.tolist() if isinstance(self.irrev, np.ndarray) else self.irrev) + [x for x in list(range(nR)) if x not in self.irrev]

		self.__dvar_mapping = dict(zip(var_index_sequence,self.__dvars))

		lin_expr_c = []
		rhs_c = []
		senses_c = ""
		names_c = []
		for compounds, rhs, sense, prefix in zip([self.__ec_to_remove, list(self.__consumed_metabs), list(self.__produced_metabs)],[0, 1, -1],['E','G','L'],['MetabRemove','MetabConsumed','MetabProduced']):
			for compound in compounds:
				vars = self.__dvar_mapping[compound]
				if isinstance(self.__dvar_mapping[compound], tuple):
					lin_expr_c.append([cplex.SparsePair(val=1, ind=vars[0]), cplex.SparsePair(val=-1, ind=vars[1])])
				else:
					lin_expr_c.append([cplex.SparsePair(val=1, ind=vars)])
				rhs_c.append(rhs)
				senses_c += sense
				names_c.append(prefix+"_"+str(compound))

		self.__model.linear_constraints.add(lin_expr=lin_expr_c, senses=senses, rhs=rhs_c, names=names_c)


		return S_full


class DualLinearSystem(IrreversibleLinearSystem):
	def __init__(self, S, irrev, T, b):
		self.__model = cplex.Cplex()
		self.__ivars = None
		self.S, self.irrev, self.T, self.b = S, irrev, T, b
		self.__c = "C"

	def get_model(self):
		return self.__model

	def get_stoich_matrix_shape(self):
		return self.S.shape

	def get_dvars(self):
		return self.__dvars

	def get_c_variable(self):
		return self.__c

	def get_dvar_mapping(self):
			return self.__dvar_mapping

	def build_problem(self):
		# Defining useful length constants
		nM, nR = self.S.shape
		# nRi, nRr = len(irrev), nR - len(irrev)
		veclens = [("u", nM), ("vp", nR), ("vn", nR), ("w", self.T.shape[0])]
		I = np.identity(nR)
		Sxi, Sxr = self.S[:, self.irrev].T, np.delete(self.S, self.irrev, axis=1).T
		Ii, Ir = I[self.irrev, :], np.delete(I, self.irrev, axis=0)
		Ti, Tr = self.T[:, self.irrev].T, np.delete(self.T, self.irrev, axis=1).T

		u, vp, vn, w = [[(pref + str(i), 0 if pref != "u" else -CPLEX_INFINITY, CPLEX_INFINITY) for i in range(n)] for
						pref, n in
						veclens]

		Sdi = np.concatenate([Sxi, Ii, -Ii, Ti], axis=1)
		Sdr = np.concatenate([Sxr, Ir, -Ir, Tr], axis=1)
		Sd = np.concatenate([Sdi, Sdr], axis=0)
		vd = chain(u, vp, vn, w)
		names, lb, ub = list(zip(*vd))
		self.__model.variables.add(names=names, lb=lb, ub=ub)

		np_names = np.array(names)
		nnz = list(map(lambda y: np.nonzero(y)[1], zip(Sd)))

		lin_expr = [cplex.SparsePair(ind=np_names[x].tolist(), val=row[x].tolist()) for row, x in zip(Sd, nnz)]
		rhs = [0] * (Sdi.shape[0] + Sdr.shape[0])
		senses = 'G' * Sdi.shape[0] + 'E' * Sdr.shape[0]
		cnames = ['Ci' + str(i) for i in range(Sdi.shape[0])] + ['Cr' + str(i) for i in range(Sdr.shape[0])]

		self.__model.linear_constraints.add(lin_expr=lin_expr, senses=senses, rhs=rhs, names=cnames)

		self.__model.variables.add(names=['C'], lb=[1], ub=[CPLEX_INFINITY])

		b_coefs = self.b.tolist() + [1]
		b_names = list(list(zip(*w))[0] + tuple(['C']))
		self.__model.linear_constraints.add(lin_expr=[(b_names, b_coefs)], senses=['L'], rhs=[0], names=['Cb'])

		vp_names = list(zip(*vp))[0]
		vn_names = list(zip(*vn))[0]

		#var_index_sequence = self.irrev + [x for x in list(range(nR)) if x not in self.irrev]
		self.__dvars = list(zip(vp_names, vn_names))
		self.__dvar_mapping = dict(zip(range(len(self.__dvars)),self.__dvars))


		return Sd

