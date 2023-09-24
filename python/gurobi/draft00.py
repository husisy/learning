import numpy as np
import scipy.sparse
import gurobipy

m = gurobipy.Model("mip1")
x = m.addVar(vtype=gurobipy.GRB.BINARY, name="x")
y = m.addVar(vtype=gurobipy.GRB.BINARY, name="y")
z = m.addVar(vtype=gurobipy.GRB.BINARY, name="z")
m.setObjective(x + y + 2 * z, gurobipy.GRB.MAXIMIZE)
m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
m.addConstr(x + y >= 1, "c1")
m.optimize()
m.getVars() #[x,y,z]
x.VarName
x.X
m.ObjVal


m = gurobipy.Model("matrix1")
x = m.addMVar(shape=3, vtype=gurobipy.GRB.BINARY, name="x")
obj = np.array([1.0, 1.0, 2.0])
m.setObjective(obj @ x, gurobipy.GRB.MAXIMIZE)
val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
row = np.array([0, 0, 0, 1, 1])
col = np.array([0, 1, 2, 0, 1])
A = scipy.sparse.csr_matrix((val, (row, col)), shape=(2, 3))
rhs = np.array([4.0, -1.0])
m.addConstr(A @ x <= rhs, name="c")
m.optimize()
x.X
m.ObjVal
