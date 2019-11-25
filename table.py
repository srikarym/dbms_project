from functools import reduce
from collections import *
from parser import *
import numpy as np

class Table:
	def __init__(self, table_name, inp_name=None, t=None):
		self.name = table_name
		if inp_name:
			arr = np.loadtxt(f"{inp_name}.txt", dtype=str, delimiter="|")
			self.t = Parser.npy_to_dict(arr)
		else:
			self.t = t

	def select(self, args, condition, name):
		if condition:
			args = ['self.t.' + arg for arg in args]
			indices = []
			for a in args:
				i = eval(f'np.where({a})')
				indices.append(i)

			if condition == 'or':
				indices = reduce(np.union1d, indices)
			else:
				indices = reduce(np.intersect1d, indices)

		else:
			args = args.replace('=', '==')
			args = 'self.t.' + args
			indices = eval(f'np.where({args})')

		new_table = {k: v[indices] for k, v in self.t.items()}
		new_table = AttrDict(new_table)

		new_table = Table(name, t=new_table)
		return new_table

	def project(self, args, name):
		new_table = {k: self.t[k] for k in args}
		new_table = AttrDict(new_table)

		new_table = Table(name, t=new_table)
		return new_table

	def avg_sum(self, args, operation='avg'):

		arr = self.t[args]
		if operation == 'sum':
			ans = np.sum(arr)
		else:
			ans = np.average(arr)
		return str(ans)

	def avg_sum_group(self, args, name, operation='avg'):
		ar1 = args[0]
		ar_group = args[1:]

		groups = [self.t[arg] for arg in ar_group]

		groups = list(zip(*groups))

		d = defaultdict(list)
		for i, g in enumerate(groups):
			d[g].append(i)

		for k, ind in d.items():
			if operation == 'avg':
				d[k] = np.mean(self.t[ar1][ind])
			else:
				d[k] = np.sum(self.t[ar1][ind])

		d = OrderedDict(sorted(d.items()))

		new_table = defaultdict(list)
		new_table[ar1] = np.array(list(d.values()))

		for key_tuple in d.keys():
			for arg, key in zip(ar_group, key_tuple):
				new_table[arg].append(key)

		new_table = Table(name, t=new_table)
		return new_table

	def sort(self, args, name):
		s = ','.join('self.t.' + a for a in args)
		to_sort = eval(f'list(zip({s}))')
		indices = Table.argsort(to_sort)

		new_table = {k: v[indices] for k, v in self.t.items()}
		new_table = Table(name, t=new_table)
		return new_table

	def moving_avg_sum(self, args, name, operation='avg'):
		col, N = args

		arr = self.t[col]

		agg = Table.moving_agg(arr, int(N), operation)

		col_name = f'mov_{operation}'

		new_table = {}

		new_table[col] = arr
		new_table[col_name] = agg
		new_table = Table(name, t=new_table)
		return new_table

	@staticmethod
	def argsort(lis):
		return sorted(range(len(lis)), key=lis.__getitem__)

	@staticmethod
	def find_operator(s):
		lis = ['==', '<=', '>=', '<', '>', '!=', '=']
		for l in lis:
			if l in s:
				return l

	@staticmethod
	def moving_agg(x, N, operation='avg'):
		ar = deque([])
		res = []
		for n in x:
			ar.append(n)
			if len(ar) == N + 1:
				ar.popleft()
			if operation == 'avg':
				res.append("{0:.4f}".format(np.mean(ar)))
			else:
				res.append(np.sum(ar))
		return res

	@staticmethod
	def join(name1, name2, args, condition, name, t1, t2):
		d1, d2 = t1.t, t2.t

		k1, k2 = list(d1.keys()), list(d2.keys())

		exec(f'{name1} = t1.t')
		exec(f'{name2} = t2.t')

		if condition:
			min_len = float('inf')
			min_index = -1
			min_lis = []

			for i, s in enumerate(args):

				lhs = ''
				rhs = ''
				if Table.find_operator(s) == '=':
					s = s.replace('=', '==')

				for key in k1:
					sub = f'{name1}.{key}'
					if sub in s:
						lhs = sub
						break

				for key in k2:
					sub = f'{name2}.{key}'
					if sub in s:
						rhs = sub
						break

				op = Table.find_operator(s)
				s = f'xx {op} yy'

				xx, yy = eval(f'np.meshgrid({lhs}, {rhs}, sparse=False, indexing="ij")')
				ind1, ind2 = eval(f'np.where({s})')

				if len(ind1) < min_len:
					min_len = len(ind1)
					min_lis = [ind1, ind2]
					min_index = i

			for i, s in enumerate(args):
				if i != min_index:
					for key in k1:
						sub = f'{name1}.{key}'
						if sub in s:
							s = s.replace(sub, sub + '[m]')
							break

					for key in k2:
						sub = f'{name2}.{key}'
						if sub in s:
							s = s.replace(sub, sub + '[n]')
							break
					ind1 = []
					ind2 = []
					for m, n in zip(*min_lis):
						if eval(s):
							ind1.append(m)
							ind2.append(n)
					min_lis = [ind1, ind2]

			indices1 = np.array(min_lis[0])
			indices2 = np.array(min_lis[1])

		else:
			s = args
			if '=' in s and ('<' not in s and '>' not in s):
				s = s.replace('=', '==')

			lhs = ''
			rhs = ''

			for key in k1:
				sub = f'{name1}.{key}'
				if sub in s:
					lhs = sub
					break

			for key in k2:
				sub = f'{name2}.{key}'
				if sub in s:
					rhs = sub

			op = Table.find_operator(s)
			s = f'xx {op} yy'

			xx, yy = eval(f'np.meshgrid({lhs}, {rhs}, sparse=False, indexing="ij")')
			indices1, indices2 = eval(f'np.where({s})')

		d1 = {f'{name1}_{k}': v[indices1] for k, v in d1.items()}
		d2 = {f'{name2}_{k}': v[indices2] for k, v in d2.items()}
		d_merged = OrderedDict({**d1, **d2})

		new_table = AttrDict(d_merged)
		new_table = Table(name, t=new_table)
		return new_table

	def __str__(self):
		keys = list(self.t.keys())
		s = ''
		n = len(self.t[keys[0]])

		max_size = [0] * len(keys)
		for i, k in enumerate(keys):
			for a in self.t[k]:
				max_size[i] = max(max_size[i], len(str(a)))

		for i in range(n):
			for j, k in enumerate(keys):
				ms = max_size[j]
				format_str = '{:<' + str(ms + 2) + '}'
				s += format_str.format(self.t[k][i])
			s += '\n'

		return s

	def __repr__(self):
		return self.__str__()