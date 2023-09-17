import casadi as cs
import numpy as np
import alpaqa

# https://kul-optec.github.io/alpaqa/develop/Sphinx/usage/getting-started.html

# Make symbolic decision variables
x1 = cs.SX.sym("x1")
x2 = cs.SX.sym("x2")
p = cs.SX.sym("p")
x = cs.vertcat(x1, x2)  # Collect decision variables into one vector

# Objective function f and the constraints function g
f = (1 - x1) ** 2 + p * (x2 - x1**2) ** 2
g = cs.vertcat(
    (x1 - 0.5) ** 3 - x2 + 1,
    x1 + x2 - 1.5,
)
# Define the bounds
C = [-0.25, -0.5], [1.5, 2.5]  # -0.25 <= x1 <= 1.5, -0.5 <= x2 <= 2.5
D = [-np.inf, -np.inf], [0, 0]  #         g1 <= 0,           g2 <= 0


problem = (alpaqa.minimize(f, x).subject_to_box(C).subject_to(g, D).with_param(p, [1])).compile()
problem.param = [10.0]
# problem.D.lowerbound[1] = -1e20

solver = alpaqa.ALMSolver(alpaqa.PANOCSolver()) #PANOC:  inner-solver
# inner_solver = alpaqa.PANOCSolver(panoc_params={'max_iter': 1000, 'stop_crit': alpaqa.PANOCStopCrit.ApproxKKT}, lbfgs_params={'memory': 10})
# tmp0 = {'tolerance': 1e-10, 'dual_tolerance': 1e-10, 'initial_penalty': 50, 'penalty_update_factor': 20,}
# solver = alpaqa.ALMSolver(alm_params=tmp0, inner_solver=inner_solver)

x0 = [0.1, 1.8]  # decision variables
y0 = [0.0, 0.0]  # Lagrange multipliers for g(x)
x_sol, y_sol, stats = solver(problem, x0, y0)
stats["status"]
problem.eval_f(x_sol)
