from functools import reduce
from collections import *
from parser import *
import numpy as np


class Table:
	"""Table.
	Base class for a table. Contains various methods which are queries that can be done on the table
	"""

	def __init__(self, table_name, file_name=None, t=None):
		"""Constructor.
			Used to create a new table instance.

			Args:
				table_name (str): Name of the table
				file_name(str): Create a table from input text file by passing file name
				t (dict): Can also be created from an existing dictionary by passing as argument t
				s (str): Input string

			Returns:
				Table() instance
		"""
		self.name = table_name
		if file_name:
			arr = np.loadtxt(f"{file_name}.txt", dtype=str, delimiter="|")
			self.t = Parser.npy_to_dict(arr)
		else:
			self.t = t

	def select(self, conditions, bool_op, name):
		"""Perform select operation on the table

			Args:
				conditions (str / list of str) : conditions to select
				Ex: ['time > 50)' , '(qty < 30)']

				bool_op (str): boolean operation separating conditions (and / or)
							None if only one condition

				name (str) : Name of new table
			Returns:
				new Table() instance after select

			Example:
				R1 = R.select( ['time > 50)' , '(qty < 30)'], 'or', 'R1')
		"""
		if bool_op:
			conditions = ['self.t.' + arg for arg in conditions]
			indices = []
			for a in conditions:
				i = eval(f'np.where({a})')
				indices.append(i)

			if bool_op == 'or':
				indices = reduce(np.union1d, indices)
			else:
				indices = reduce(np.intersect1d, indices)

		else:
			conditions = conditions.replace('=', '==')
			conditions = 'self.t.' + conditions
			indices = eval(f'np.where({conditions})')

		new_table = {k: v[indices] for k, v in self.t.items()}
		new_table = AttrDict(new_table)

		new_table = Table(name, t=new_table)
		return new_table

	def project(self, elements, name):

		""" Perform project operation on the table

			Args:
				elements (str / list of str) : Columns to project from source table
				Ex: ['saleid', 'qty', 'pricerange']

				name (str) : Name of new table
			Returns:
				new Table() instance after project

			Example:
				R2 := R1.project( ['saleid', 'qty', 'pricerange'], 'R2')
		"""
		new_table = {k: self.t[k] for k in elements}
		new_table = AttrDict(new_table)

		new_table = Table(name, t=new_table)
		return new_table

	def avg_sum(self, elements, operation='avg'):

		""" Perform average / sum on the table

			Args:
				elements (str) : Source table Column to calculate average / sum
				Ex: 'qty'

			Returns:
				Float - average / sum

			Example:
				R3 = R1.avg('qty')
		"""

		arr = self.t[elements]
		if operation == 'sum':
			ans = np.sum(arr)
		else:
			ans = np.average(arr)
		return str(ans)

	def avg_sum_group(self, args, name, operation='avg'):
		""" Perform average / sum group on the table

			Args:
				args (str) : list of strings.
							1st element - source table Column to calculate average / sum
							rest of the list - columns to group by

				Ex: ['qty','time','percentage']
				name (str) : Name of new table
			Returns:
				new Table() instance after avg/sum group

			Example:
				R5 := R1.avg_sum_group( ['qty','time','percentage'] , 'R5','sum')
		"""

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
		""" Sort a table by columns

			Args:
				args (str) : list of strings to sort by
				Ex: ['R1_time', 'S_C']
				name (str) : Name of new table
			Returns:
				new Table() instance after sorting

			Example:
				T2prime := T1.sort(['R1_time', 'S_C'], 'T2prime')
		"""
		s = ','.join('self.t.' + a for a in args)
		to_sort = eval(f'list(zip({s}))')
		indices = Table.argsort(to_sort)

		new_table = {k: v[indices] for k, v in self.t.items()}
		new_table = Table(name, t=new_table)
		return new_table

	def moving_avg_sum(self, args, name, operation='avg'):
		""" Perform moving average / sum on the table based on operation

			Args:
				elements (list) :
						1st element: Source table Column to calculate average / sum
						2nd element: window size (N)

			Returns:
				new Table() instance after moving average / sum

			Example:
				T3 := T2prime.moving_avg_sum( ['R1_qty', 3],'T3', 'avg')
		"""

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
	def argsort(arr):
		""" Returns the indices that would sort an array.
			Also works on array of tuples

			Args:
				arr (list) : Array to sort.

			Returns:
				index_array : Array of indices that sort arr

			Example:
				x = [3,2,4,1,6]
				>>Table.argsort(x)
				[3, 1, 0, 2, 4]
		"""
		return sorted(range(len(arr)), key=arr.__getitem__)

	@staticmethod
	def find_operator(s):
		""" Returns the operator in a string.

			Args:
				s (str) : Input string.

			Returns:
				string : Operator

			Example:
				s = 'A.col1 >= B.col2'
				>>Table.find_operator(s)
				'>='
		"""
		lis = ['==', '<=', '>=', '<', '>', '!=', '=']
		for l in lis:
			if l in s:
				return l

	@staticmethod
	def moving_agg(x, N, operation='avg'):
		""" Calculates the moving average / sum on an array.

			Args:
				x (list) : Input array of integers / float.
				N (int) : Window size

			Returns:
				list : Moving averages / sums

			Example:
				arr = [4, 8, 9, 7]
				>>Table.moving_agg(arr)
				[4,6,7,8]
		"""

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

	@classmethod
	def join(cls,name1, name2, args, bool_op, name, t1, t2):

		""" Joins two tables based on conditions

			Args:
				name1 (str) : Name of table1
				name2 (str) : Name of table2

				args (str / list of str): Conditions on table columns
				Ex: ['(R1.qty > S.Q)' , ' (R1.saleid = S.saleid)']

				bool_op (str): boolean operation separating conditions (usually and)
							None if only one condition is present

				name (str): Name of output table

				t1 (Table) : table1 instance
				t2 (Table) : table2 instance

			Returns:
				new Table() instance after moving average / sum

			Example:
				T1 := Table.join('R1', 'S', ['(R1.qty > S.Q)' , ' (R1.saleid = S.saleid)'] ,
									'T1', R1, S)
		"""
		d1, d2 = t1.t, t2.t

		k1, k2 = list(d1.keys()), list(d2.keys())

		exec(f'{name1} = t1.t')
		exec(f'{name2} = t2.t')

		if bool_op:
			min_len = float('inf')
			min_index = -1
			min_lis = []

			for i, s in enumerate(args):

				lhs = ''
				rhs = ''
				if cls.find_operator(s) == '=':
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

				op = cls.find_operator(s)
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

			op = cls.find_operator(s)
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

		""" Called by the str() function and by the print statement
			to compute the “informal” string representation of a Table object

			Each row has left aligned elements and rows are separated by \n newline

			Returns:
				string : string representation of the object

			Example:
				d = {'Name': array(['a', 'c', 'e'], dtype='<U4'),
				 'ID': array([0, 1, 2]),
				 'Qty': array([12, 15, 13])}

				table1 = Table(name = 'sample', t = d)

				print(table1)

				a  0  12
				c  1  15
				e  2  13

			"""
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
		#Same as __str__()

		return self.__str__()