import pytosolver as opt
from pytosolver.solvers.abstractsolverapi import AbstractSolverApi


def get_problem(solver_api: type[AbstractSolverApi]):
    """
    Max: -(x1 + x2 + 5)\\
    s.t.\\
    x2 - x1 >= 2\\
    x1 + x2 >= 0\\
    x1 in R\\
    x2 in R
    """
    x1 = opt.Variable('x1')
    x2 = opt.Variable('x2')

    constr1 = opt.LinearConstraint(x2 - x1 >= 2, 'constr1')
    constr2 = opt.LinearConstraint(x1 + x2 >= 0, 'constr2')

    objective_function = -(x1 + x2 + 5)

    prob = (
        opt.Problem(name="minimal_problem", solver_api=solver_api)
        .add_vars(x1, x2)
        .add_constrs(constr1, constr2)
        .set_objective(opt.ObjectiveFunction(objective_function, is_minimization=False))
    )

    return prob, x1, x2, constr1, constr2

def main():
    from parse_solver import get_solver_api

    prob, x1, x2, *_ = get_problem(get_solver_api())

    print(f"\n{prob}\n")

    prob.solve()
    print("Solve status:", prob.solve_status)
    print("Objective function value:", prob.get_objectivefunction_value())

    print("x1:", x1.value)
    print("x2:", x2.value)

if __name__ == '__main__':
    main()
