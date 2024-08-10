import sys

from pytosolver.solvers.highsapi import HighsApi
from pytosolver.solvers.xpressapi import XpressApi
from pytosolver.solvers.abstractsolverapi import AbstractSolverApi


def get_solver_api() -> type[AbstractSolverApi]:
    solver_name = None if len(sys.argv) == 1 else str(sys.argv[1]).lower()
    if solver_name is None:
        return HighsApi

    match solver_name:
        case "xpress":
            return XpressApi
        case "highs":
            return HighsApi
        case _:
            return HighsApi
