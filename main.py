from myparser import Parser
from utils import Timer
from table import Table

net_id = 'msy290_sd3770'
query_file = 'queries.txt'


def safe_run(func):
    """
        Decorator that handles errors
    """
    def func_wrapper(*args, **kwargs):

        try:
            return func(*args, **kwargs)
        except Exception as e:

            print(e)
            return None

    return func_wrapper


class Solution:
    """
        Stores various tables in memory as class variables
    """
    f = open(f"{net_id}_AllOperations.txt", "w")

    @classmethod
    @safe_run
    def read_line(cls, line):
        """
            Parses the string and calls the method from table class
        """
        line = Parser.remove_comments(line)
        if line:
            with Timer() as t:
                if ':=' in line:
                    left, right = Parser.splitlr(line)

                    if 'inputfromfile' in right:
                        inp_name = Parser.parse(right, 'input')
                        output = Table(left, inp_name)

                    elif 'select' in right:
                        base_table, args, condition = Parser.parse(
                            right, 'select')

                        output = getattr(
                            cls, base_table).select(
                            args, condition, left)

                    elif 'project' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(cls, base_table).project(args, left)

                    elif 'avggroup' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(
                            cls, base_table).avg_sum_count_group(
                            args, left)

                    elif 'movavg' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(
                            cls, base_table).moving_avg_sum(
                            args, left)
                        setattr(cls, left, output)

                    elif 'avg' in right:
                        base_table, args = Parser.parse(right, 'avg')

                        output = getattr(cls, base_table).avg_sum_count(args)

                    elif 'sumgroup' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(
                            cls, base_table).avg_sum_count_group(
                            args, left, 'sum')

                    elif 'movsum' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(
                            cls, base_table).moving_avg_sum(
                            args, left, 'sum')

                    elif 'sum' in right:
                        base_table, args = Parser.parse(right, 'avg')

                        output = getattr(
                            cls, base_table).avg_sum_count(
                            args, 'sum')

                    elif 'sort' in right:
                        base_table, args = Parser.parse(right, 'project')
                        output = getattr(cls, base_table).sort(args, left)

                    elif 'join' in right:
                        name1, name2, args, condition = Parser.parse(
                            right, 'join')

                        table1 = getattr(cls, name1)
                        table2 = getattr(cls, name2)

                        output = Table.join(
                            name1, name2, args, condition, left, table1, table2)

                    elif 'concat' in right:
                        args = Parser.parse(right, 'concat')

                        tables = []
                        for a in args:
                            tables.append(getattr(getattr(cls, a), 't'))
                        output = Table.concat(tables, left)

                    elif 'countgroup' in right:
                        base_table, args = Parser.parse(right, 'project')

                        output = getattr(
                            cls, base_table).avg_sum_count_group(
                            args, left, 'count')

                    elif 'count' in right:
                        base_table, args = Parser.parse(right, 'avg')

                        output = getattr(
                            cls, base_table).avg_sum_count(
                            args, 'count')

                    setattr(cls, left, output)

                    cls.f.write(line + '\n')
                    cls.f.write(str(output))
                    cls.f.write('\n')

                else:
                    base_table, col = Parser.parse(line, 'Btree')
                    table = getattr(cls, base_table)
                    if line.startswith('Btree'):
                        table.index_btree(col)

                    elif line.startswith('Hash'):
                        table.index_hash(col)

                    elif line.startswith('outputtofile'):
                        table.output(col)

            print(line, '\nQuery took %.06f sec.\n' % t.interval)


if __name__ == '__main__':

    with open(query_file, 'r') as f:
        for i, line in enumerate(f):
            Solution.read_line(line)
    Solution.f.close()
