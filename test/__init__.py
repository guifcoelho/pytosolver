from unittest import TestCase

from pytosolver import Problem


class Test(TestCase):
    def set_log(self, problem: Problem, solver_api_name: str, flag: bool):
        if solver_api_name == 'highs':
            problem.set_option('output_flag', flag)
        elif solver_api_name == 'xpress':
            problem.set_option('OUTPUTLOG', 1 if flag else 0)