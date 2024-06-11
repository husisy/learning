import numpy as np

import gekko

model = gekko.GEKKO()
x0 = model.Array(model.Var, 4, value=1, lb=1, ub=5)
# change guess
x0[1].value = 5
x0[2].value = 5
model.Equation(np.prod(x0)>=25)# prod>=25
model.Equation(model.sum([xi**2 for xi in x0])==40) #sum=40
model.Minimize(x0[0] * x0[3] * (x0[0]+x0[1]+x0[2]) + x0[2]) #objective
# model.options.IMODE = 3 #steady state optimization (optional)
model.solve()
# model.path #path to gekko working directory
x0 #(np,object)
np0 = np.array([xi.value for xi in x0])
