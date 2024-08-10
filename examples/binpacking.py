"""
Binpacking example taken from `https://github.com/ERGO-Code/HiGHS/blob/master/examples/branch-and-price.py`
"""

from collections import defaultdict
from operator import itemgetter
import random

import pytosolver as opt
from pytosolver.solvers.abstractsolverapi import AbstractSolverApi


SEED = 100
random.seed(SEED)

NumberItems = 100
ItemWeights = [round(random.uniform(1, 10), 1) for _ in range(NumberItems)]
BinCapacity = 15

def solveGreedyModel():
    bins = defaultdict(float)
    solution = defaultdict(list)

    for item, w in sorted(enumerate(ItemWeights), reverse=True, key=itemgetter(1)):
        index = next((i for i, W in bins.items() if W + w <= BinCapacity), len(bins))
        bins[index] += w
        solution[index].append(item)

    return list(solution.values())

def get_problem(solver_api: type[AbstractSolverApi]):
    greedy_bins = solveGreedyModel()
    B = len(greedy_bins)

    prob = opt.Problem(name='BinPacking', solver_api=solver_api)

    x = {
        (i,j): opt.Variable(f"x({i},{j})", vartype=opt.VarType.BIN)
        for i in range(NumberItems)
        for j in range(B)
    }
    y = {
        j: opt.Variable(f"y({j})", vartype=opt.VarType.BIN)
        for j in range(B)
    }
    prob.add_vars(x, y)

    prob.set_objective(opt.Sum(y))

    for i in range(NumberItems):
        prob.add_constr(opt.Sum(x[i,j] for j in range(B)) == 1)

    for j in range(B):
        prob.add_constr(opt.Sum(ItemWeights[i] * x[i,j] for i in range(NumberItems)) <= BinCapacity*y[j])

    return prob, x, y

def main():
    from parse_solver import get_solver_api

    greedy_bins = solveGreedyModel()
    B = len(greedy_bins)

    prob, x, y = get_problem(get_solver_api())

    # Hot starting variables
    for j, var_y in y.items():
        var_y.set_hotstart(1.0)

    for (i,j), var_x in x.items():
        if i in greedy_bins[j]:
            var_x.set_hotstart(1.0)

    prob.solve(with_hotstart=True)

    if prob.solve_status not in [opt.SolveStatus.FEASIBLE, opt.SolveStatus.OPTIMUM]:
        print("The model is not feasible")
        exit()

    bins = [
        [i for i in range(NumberItems) if x[i,j].value or 0 > 0.9]
        for j in range(B)
        if y[j].value or 0 > 0.9
    ]

    for bin, items in enumerate(bins):
        tt_weight = round(sum(ItemWeights[i] for i in items))
        print(f"Bin {bin+1} ({tt_weight} <= {BinCapacity}): {items}")
        assert tt_weight <= BinCapacity

if __name__ == '__main__':
    main()
