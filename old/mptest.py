if __name__ == '__main__':
	import multiprocessing
	def add1(x):
		return x+1
	pool = multiprocessing.Pool(10)

	res = pool.map(print, range(100000000))