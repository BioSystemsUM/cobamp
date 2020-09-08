from functools import partial


class CommandHistory(object):
	def __init__(self):
		self.commands = []

	def execute_last(self):
		self.commands.pop()()

	def execute_first(self):
		self.commands.pop(0)()

	def execute_all(self, forward=False):
		while len(self.commands) > 0:
			(self.execute_first if forward else self.execute_last)()

	def queue_command(self, func, args):
		self.commands.append(partial(func, **args))