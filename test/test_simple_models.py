import pytosolver
import examples.minimal as minimal_example
import examples.multiobjective as multiobjective_example
from test.env_solver_api import get_solver_api
from test import Test


SOLVER_API, SOLVER_API_NAME = get_solver_api()

class SimpleModelsTest(Test):

    def test_minimial_example_model(self):
        prob, x1, x2, constr1, constr2 = minimal_example.get_problem(SOLVER_API)

        self.set_log(prob, SOLVER_API_NAME, False)

        prob.solve()
        self.assertTrue(prob.solve_status == pytosolver.SolveStatus.OPTIMUM)
        self.assertTrue(x1.value, -1)
        self.assertTrue(x2.value, 1)

    def test_multiobjective_example_model(self):
        prob, variables, constraints = multiobjective_example.get_problem(SOLVER_API)
        self.set_log(prob, SOLVER_API_NAME, False)
        prob.solve()
        self.assertTrue(prob.solve_status == pytosolver.SolveStatus.OPTIMUM)