import numpy as np

import pysat
import pysat.solvers
import pysat.formula

# satisfiable
g = pysat.solvers.Glucose3()
g.add_clause([-1, 2]) # !x1 | x2
g.add_clause([-2, 3]) # !x2 | x3
g.solve()
x = g.get_model() # [-1,-2,-3] (x1,x2,x3)=(False,False,False)


# unsatisfiable
with pysat.solvers.Minisat22(bootstrap_with=[[-1, 2], [-2, 3]]) as m:
    print(m.solve(assumptions=[1, -3])) #False
    print(m.get_core()) #[-3, 1]


# with proof
# from pysat.formula import CNF
# from pysat.solvers import Lingeling

formula = pysat.formula.CNF()
formula.append([-1, 2])
formula.append([1, -2])
formula.append([-1, -2])
formula.append([1, 2])

with pysat.solvers.Lingeling(bootstrap_with=formula.clauses, with_proof=True) as l:
    if l.solve() == False:
        print(l.get_proof()) # ['2 0', '1 0', '0']
