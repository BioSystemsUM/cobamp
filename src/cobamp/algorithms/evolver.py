from inspyred import ec
from random import Random
from numpy import nan, array
from cobamp.algorithms.kshortest import KShortestEnumerator
from cobamp.core.linear_systems import IrreversibleLinearSystem
import logging

class EFMEvolver(object):
	'''
	Implements the approach of Kaleta et al. described in their 2009 conference paper: 'EFMEvolver: Computing Elementary
	 Flux Modes in Genome-scale Metabolic Networks.'
	'''
	def __init__(self, linear_system, evolutionary_computation):

		logger = logging.getLogger('inspyred.ec')
		logger.setLevel(logging.DEBUG)
		file_handler = logging.FileHandler('inspyred.log', mode='w')
		file_handler.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

		self.enum = KShortestEnumerator(linear_system)
		self.ind_len = len(linear_system.get_dvar_mapping())
		self.rng = Random()
		self.ea = evolutionary_computation(self.rng)
			# 'generator': self.generate,
		self.ea.variator = [ec.variators.uniform_crossover,ec.variators.bit_flip_mutation]
		self.ea.replacer = self.replacer
		self.ea.terminator = ec.terminators.evaluation_termination
		self.ea.selector = self.selector

		self.ind_generator = ec.generators.strategize(self.generate)

	def replacer(self, random, population, parents, offspring, args):
		psize = len(population)
		## TODO: Use multiprocessing for MILPs
		survivors = []
		for individual in offspring:
			viable, sol = self.verify_viability(individual)
			if viable:
				print('Found EFM')
				individual.candidate[:self.ind_len] = sol
			survivors.append(individual)

		# num_remaining = psize - len(survivors)
		# for i in range(num_remaining):
		# 	survivors.append(random.choice(l))

		for i in range(psize-len(survivors)):
			survivors.append(ec.Individual(candidate=self.ind_generator(random, args)))

		return survivors

	def generate(self, random, args):
		size = args.get('num_inputs', self.ind_len)
		return [random.randint(0, 1) for _ in range(size)]

	def selector(self, random, population, args):
		population_fitness = sum([self.evaluate([p],args)[0] for p in population])
		selected = []
		for ind in population:
			select = (self.evaluate([ind], args)[0]/population_fitness - random.random()) > 0
			if select:
				selected.append(ind)

		return selected

	def verify_viability(self, efm):
		## TODO: Implement MILP version of this
		if isinstance(efm, ec.Individual):
			true_efm = efm.candidate[:self.ind_len]
		else:
			true_efm = efm[:self.ind_len]
		self.enum.reset_enumerator_state()
		self.enum.force_solutions([[i for i, e in enumerate(true_efm) if e > 0]])
		self.enum.set_size_constraint(1)
		sol = self.enum.get_single_solution()
		if hasattr(sol, '_Solution__obj_value') and sol.objective_value() != nan:
			ind_sol = self.individual_from_sol(sol)
			return True, ind_sol
		else:
			return False, []

	def evaluate(self, cands, args):
		pop_condensed = list(zip(*[c.candidate[:self.ind_len] if isinstance(c, ec.Individual) else c[:self.ind_len] for c in args['_ec'].population]))
		pop_sum = [sum(l) for l in pop_condensed]

		return [sum([(s/p) if p > 0 else 0 for s, p in zip(cand.candidate[:self.ind_len] if isinstance(cand, ec.Individual) else cand[:self.ind_len], pop_sum)]) for cand in cands]

	def individual_from_sol(self, sol):
		ind = [0] * self.ind_len
		for i in sol.get_active_indicator_varids():
			ind[i] = 1
		return ind

	def evolve(self, **evolve_args):
		return self.ea.evolve(
			evaluator=self.evaluate,
			generator=self.generate,
			num_inputs=self.ind_len,
			**evolve_args
		)


if __name__ == '__main__':
	S = array([[1, -1, 0, 0, -1, 0, -1, 0, 0],
					   [0, 1, -1, 0, 0, 0, 0, 0, 0],
					   [0, 1, 0, 1, -1, 0, 0, 0, 0],
					   [0, 0, 0, 0, 0, 1, -1, 0, 0],
					   [0, 0, 0, 0, 0, 0, 1, -1, 0],
					   [0, 0, 0, 0, 1, 0, 0, 1, -1]])
	rx_names = ["R" + str(i) for i in range(1, 10)]
	irrev = [0, 1, 2, 4, 5, 6, 7, 8]

	T = array([0] * S.shape[1]).reshape(1, S.shape[1])
	T[0, 8] = -1
	b = array([-1]).reshape(1, )

	evolver = EFMEvolver(linear_system=IrreversibleLinearSystem(S, irrev), evolutionary_computation=ec.ES)
	f = evolver.evolve(pop_size=100, max_evaluations=1000)

	evolver.ea