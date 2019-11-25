from collections import defaultdict, OrderedDict
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


class OrderedDefaultDict(OrderedDict):
	factory = list

	def __missing__(self, key):
		self[key] = value = self.factory()
		return value


class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self