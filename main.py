import os
import sys
from parser import *
from utils import *

class Solution:
	def __init__(self):
		pass

	def read_line(self, line):

		line = Parser.remove_comments(line)
		if line:
			with Timer() as t:
				if ':=' in line:
					left, right = Parser.splitlr(line)
					if 'inputfromfile' in right:
						inp_name = Parser.parse(right, 'input')
						exec(f'self.{left} = Table(left,inp_name)')

					elif 'select' in right:
						base_table, args, condition = Parser.parse(right, 'select')
						exec(f'self.{left} = self.{base_table}.select(args,condition,left)')

					elif 'project' in right:
						base_table, args = Parser.parse(right, 'project')
						exec(f'self.{left} = self.{base_table}.project(args,left)')

					elif 'avggroup' in right:
						base_table, args = Parser.parse(right, 'project')
						exec(f'self.{left} = self.{base_table}.avg_sum_group(args,left)')

					elif 'movavg' in right:
						base_table, args = Parser.parse(right, 'project')
						exec(f'self.{left} = self.{base_table}.moving_avg_sum(args,left)')

					elif 'avg' in right:
						base_table, args = Parser.parse(right, 'avg')
						exec(f'self.{left} = self.{base_table}.avg_sum(args)')

					elif 'sumgroup' in right:
						base_table, args = Parser.parse(right, 'project')
						exec(f'self.{left} = self.{base_table}.avg_sum_group(args,left,"sum")')

					elif 'movsum' in right:
						base_table, args = Parser.parse(right, 'project')
						exec(f'self.{left} = self.{base_table}.moving_avg_sum(args,left,"sum")')

					elif 'sum' in right:
						base_table, args = Parser.parse(right, 'avg')
						exec(f'self.{left} = self.{base_table}.avg_sum(args,sum)')

					elif 'sort' in right:
						base_table, args = Parser.parse(right, 'project')
						exec(f'self.{left} = self.{base_table}.sort(args,left)')
					elif 'join' in right:
						t1, t2, args, condition = Parser.parse(right, 'join')
						exec(f'self.{left} = Table.join(t1,t2,args,condition,left,{"self." + t1},{"self." + t2})')

			print(line, '\nQuery took %.06f sec.\n' % t.interval)

if __name__ == 'main':
	query_file = 'queries.txt'
	solution = Solution()
	with open(query_file, 'r') as f:
		for i, line in enumerate(f):
			solution.read_line(line)