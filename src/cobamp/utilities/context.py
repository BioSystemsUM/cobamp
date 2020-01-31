from functools import partial


class CommandHistory(object):
	def __init__(self):
		self.commands = []

	def execute_last(self):
		self.commands.pop(len(self.commands) - 1)()

	def execute_first(self):
		self.commands.pop(len(self.commands) - 1)()

	def execute_all(self, forward=False):
		fx = self.execute_first if forward else self.execute_last
		while len(self.commands) > 0:
			fx()

	def queue_command(self, func, args):
		self.commands.append(partial(func, **args))