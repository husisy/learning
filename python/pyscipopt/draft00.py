import numpy as np
import pyscipopt

model = pyscipopt.Model("Example") #name is optional
x = model.addVar("x")
y = model.addVar("y", vtype="INTEGER")
model.setObjective(x + y)
model.addCons(2*x - y*y >= 0)
model.optimize()
sol = model.getBestSol()
sol[x]
sol[y]
