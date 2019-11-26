from utils import *

class Parser:
	#Class for parser. Has various class methods

	def __init__(self):
		pass

	@classmethod
	def remove_comments(cls, s):
		"""Removes everything after //

		Args:
			s (str): Input string
		Returns:
			String without comments and newline characters
		"""
		if '//' in s:
			s = s[:s.find('//')]  # Remove comments
		s = s.strip()
		return s



	@classmethod
	def splitlr(cls, s):
		"""Splits a line which has := into two parts (lhs and rhs of :=).

		Args:
			s (str): Input string
		Returns:
			Tuple of strings: left, right which are left and right substrings of := excluding :=

		Examples:
			s = 'R := inputfromfile(sales1)'
			>> Parser.splitlr(s)
			('R', 'inputfromfile(sales1)')

		"""
		left, right = s.split(':=')
		left = left.strip()
		right = right.strip()
		return left, right


	@classmethod
	def remove_outer_paren(cls, s):
		"""Removes outermost parenthesis of a string.
		Used on the right part after splitlr.
		Doesnt work when nested parenthesis are present
		Args:
			s (str): Input string
		Returns:
			String without outer ()

		Examples:
			s = 'project(R1, saleid, qty, pricerange) '
			>> Parser.remove_outer_paren(s)
			'R1, saleid, qty, pricerange'

		"""
		s = s[s.find('(') + 1:s.find(')')]
		return s


	@classmethod
	def parse(cls, s, call):

		"""Parses the string based on the query
		Args:
			s (str): Input string
		Returns:
			Tuple of strings depending on the query

		Examples:
			s = 'select(R, (time > 50) or (qty < 30))'
			>> Parser.parse(s,'select')
			('R', ['time > 50', 'qty < 30'], 'or')

			s = 'project(R1, saleid, qty, pricerange) '
			>> Parser.parse(s,'project')
			('R1', ['saleid', 'qty', 'pricerange'])

			s = 'avg(R1, qty) '
			>> Parser.parse(s,'avg')
			('R1', 'qty')

		"""
		if call == 'input':
			return cls.remove_outer_paren(s)

		elif call == 'select':
			s = s[s.find("(") + 1:s.rfind(")")]
			base_table, args = s.split(',')
			base_table = base_table.strip()
			args = args.strip()

			if 'or' not in args and 'and' not in args:
				return base_table, args, None
			condition = 'or' if 'or' in args else 'and'

			args = args.split(condition)
			args = [cls.remove_outer_paren(s.strip()) for s in args]
			return base_table, args, condition

		elif call == 'project':
			s = s[s.find("(") + 1:s.rfind(")")]
			base_table, args = s.split(',')[0], s.split(',')[1:]
			base_table = base_table.strip()
			args = [a.strip() for a in args]

			return base_table, args

		elif call == 'avg':
			s = s[s.find("(") + 1:s.rfind(")")]
			base_table, args = s.split(',')[0], s.split(',')[1]
			base_table = base_table.strip()
			args = args.strip()
			return base_table, args


		elif call == 'join':
			s = s[s.find("(") + 1:s.rfind(")")]
			t1, t2, args = s.split(',')[0], s.split(',')[1], s.split(',')[2:]
			t1 = t1.strip()
			t2 = t2.strip()
			args = [a.strip() for a in args]

			if ' and ' not in args[0]:
				return t1, t2, args[0].strip(), None

			condition = 'and'
			args = ','.join(args).split(condition)
			args = [cls.remove_outer_paren(s.strip()) for s in args]
			return t1, t2, args, condition


	@classmethod
	def npy_to_dict(cls, arr):
		"""Converts 2d numpy array to dictionary

		Elements of the first row are keys of the dictionary
		Rest of the elements of every column are its corresponding values

		Args:
			arr (array): Input array (2d)
		Returns:
			dictionary of arrays

		Examples:
			a = np.array([['Name','ID', 'Qty'],['a',0,12], ['c',1,15],['e',2,13]])
			>> Parser.npy_to_dict(a)
			{'Name': array(['a', 'c', 'e'], dtype='<U4'),
			 'ID': array([0, 1, 2]),
			 'Qty': array([12, 15, 13])}
		"""
		D = AttrDict()
		keys = arr[0]
		for i, k in enumerate(keys):
			try:
				D[k] = arr[1:, i].astype(int)
			except:
				D[k] = arr[1:, i]
		return D
