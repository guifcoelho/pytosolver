import os

def get_solver_api():
    solver_api_name = os.getenv('SOLVER_API', 'highs').lower()
    match solver_api_name:
        case "xpress":
            from pytosolver.solvers.xpressapi import XpressApi
            return XpressApi, solver_api_name
        case "highs":
            from pytosolver.solvers.highsapi import HighsApi
            return HighsApi, solver_api_name
        case _:
            from pytosolver.solvers.highsapi import HighsApi
            return HighsApi, solver_api_name