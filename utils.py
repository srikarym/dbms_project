import time
class Timer:
	"""With block timer.
	with Timer() as t:
		foo = blah()
	print('Request took %.03f sec.' % t.interval)
	"""

	def __enter__(self):
		self.start = time.perf_counter()
		return self

	def __exit__(self, *args):
		self.end = time.perf_counter()
		self.interval = self.end - self.start


class AttrDict(dict):

	"""Converts a dictionary to attribute dictionary
	keys of the original dictionary can be accessed as attributes of the new dictionary

	Args:
		d (dict): Input dictionary

	Examples:
		d = {'key1':[1,2,3], 'key2':[4,5,6]}
		d1 = AttrDict(d)

		>>print(d1.key1)
		[1,2,3]
	"""
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self