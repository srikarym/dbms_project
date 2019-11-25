from utils import *

class Parser:
	def __init__(self):
		pass

	@classmethod
	def remove_comments(cls, s):
		if '//' in s:
			s = s[:s.find('//')]  # Remove comments
		s = s.strip()
		return s

	@classmethod
	def splitlr(cls, s):
		left, right = s.split(':=')
		left = left.strip()
		right = right.strip()
		return left, right

	@classmethod
	def remove_outer_paren(cls, s):
		s = s[s.find('(') + 1:s.find(')')]
		return s

	@classmethod
	def parse(cls, s, call):
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
		D = AttrDict()
		keys = arr[0]
		for i, k in enumerate(keys):
			try:
				D[k] = arr[1:, i].astype(int)
			except:
				D[k] = arr[1:, i]
		return D
