import os
import sys
from myparser import *
from utils import *
from table import *


class Solution:

    @classmethod
    def read_line(cls, line):

        line = Parser.remove_comments(line)
        if line:
            with Timer() as t:
                if ':=' in line:
                    left, right = Parser.splitlr(line)

                    if 'inputfromfile' in right:
                        inp_name = Parser.parse(right, 'input')
                        setattr(cls, left, Table(left, inp_name))

                    elif 'select' in right:
                        base_table, args, condition = Parser.parse(
                            right, 'select')

                        output = getattr(cls,base_table).select(args,condition,left)
                        setattr(cls, left, output)

                    elif 'project' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(cls, base_table).project(args,left)
                        setattr(cls, left, output)

                    elif 'avggroup' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(cls, base_table).avg_sum_group(args,left)
                        setattr(cls, left, output)

                    elif 'movavg' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(cls, base_table).moving_avg_sum(args,left)
                        setattr(cls, left, output)

                    elif 'avg' in right:
                        base_table, args = Parser.parse(right, 'avg')

                        output = getattr(cls, base_table).avg_sum(args)
                        setattr(cls, left, output)

                    elif 'sumgroup' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(cls, base_table).avg_sum_group(args, left, 'sum')
                        setattr(cls, left, output)

                    elif 'movsum' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(cls, base_table).moving_avg_sum(args, left, 'sum')
                        setattr(cls, left, output)

                    elif 'sum' in right:
                        base_table, args = Parser.parse(right, 'avg')

                        output = getattr(cls, base_table).avg_sum(args,sum)
                        setattr(cls, left, output)

                    elif 'sort' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(cls,base_table).sort(args,left)
                        setattr(cls, left, output)

                    elif 'join' in right:
                        name1, name2, args, condition = Parser.parse(right, 'join')

                        table1 = getattr(cls, name1)
                        table2 = getattr(cls, name2)

                        output = Table.join(name1,name2,args, condition, left, table1, table2)
                        setattr(cls, left, output)

                    elif 'concat' in right:
                        args = Parser.parse(right,'concat')

                        tables = []
                        for a in args:
                            tables.append(getattr(getattr(cls, a), 't'))
                        output = Table.concat(tables, left)

                        setattr(cls, left, output)

                else:
                    if line.startswith('Btree'):
                        base_table, col = Parser.parse(line, 'Btree')
                        table = getattr(cls, base_table)
                        table.index_btree(col)

                    elif line.startswith('Hash'):
                        base_table, col = Parser.parse(line, 'Btree')
                        table = getattr(cls, base_table)
                        table.index_hash(col)

            print(line, '\nQuery took %.06f sec.\n' % t.interval)


if __name__ == '__main__':
    query_file = 'queries.txt'
    with open(query_file, 'r') as f:
        for i, line in enumerate(f):
            Solution.read_line(line)
