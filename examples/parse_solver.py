import sys

from pytosolver.solvers.abstractsolverapi import AbstractSolverApi


def get_solver_api() -> type[AbstractSolverApi]:
    solver_name = None if len(sys.argv) == 1 else str(sys.argv[1]).lower()

    match solver_name:
        case "xpress":
            from pytosolver.solvers.xpressapi import XpressApi
            return XpressApi
        case "highs":
            from pytosolver.solvers.highsapi import HighsApi
            return HighsApi
        case _:
            from pytosolver.solvers.highsapi import HighsApi
            return HighsApi
