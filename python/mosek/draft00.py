import sys #M.setLogHandler(sys.stdout)
import numpy as np

import mosek
import mosek.fusion

def mosek_linear_equal(M, x, A, b):
    assert isinstance(A, np.ndarray) and isinstance(b, np.ndarray)
    for ind0 in range(A.shape[0]):
        M.constraint(f'eq{ind0}', mosek.fusion.Expr.dot(A[ind0].tolist(), x), mosek.fusion.Domain.equalsTo(b[ind0].item()))

def mosek_linear_less_equal(M, x, A, b):
    assert isinstance(A, np.ndarray) and isinstance(b, np.ndarray)
    for ind0 in range(A.shape[0]):
        M.constraint(f'leq{ind0}', mosek.fusion.Expr.dot(A[ind0].tolist(), x), mosek.fusion.Domain.lessThan(b[ind0].item()))

# linear optimization
# https://docs.mosek.com/latest/pythonfusion/tutorial-lo-shared.html#doc-tutorial-lo
with mosek.fusion.Model("lo1") as M:
    Aeq = np.array([[3, 1, 2, 0]])
    beq = np.array([30])
    Aleq = np.array([[-1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,-1], [0,1,0,0], [-2,-1,-3,-1], [0,2,0,3]])
    bleq = np.array([0,0,0,0,10,-15,25])
    c = [3.0, 1.0, 5.0, 1.0]
    x = M.variable("x", 4)
    mosek_linear_equal(M, x, Aeq, beq)
    mosek_linear_less_equal(M, x, Aleq, bleq)
    M.objective("obj", mosek.fusion.ObjectiveSense.Maximize, mosek.fusion.Expr.dot(c, x))
    M.solve()
    sol = x.level() #(np,float64,4)
    assert np.abs(sol-np.array([0,0,15,25/3])).max() < 1e-7


# conic quadratic optimization
# https://docs.mosek.com/latest/pythonfusion/tutorial-cqo-shared.html
with mosek.fusion.Model('cqo1') as M:
    x = M.variable('x', 3, mosek.fusion.Domain.greaterThan(0))
    y = M.variable('y', 3)

    M.constraint("lc", mosek.fusion.Expr.dot([1,1,2], x), mosek.fusion.Domain.equalsTo(1))

    # Create the constraints
    #      z1 belongs to C_3
    #      z2 belongs to K_3
    # where C_3 and K_3 are respectively the quadratic and
    # rotated quadratic cone of size 3, i.e.
    #                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
    #  and  2.0 z2[0] z2[1] >= z2[2]^2
    z1 = mosek.fusion.Var.vstack(y.index(0), x.slice(0, 2)) #y[0] x[0] x[1]
    z2 = mosek.fusion.Var.vstack(y.slice(1, 3), x.index(2)) #y[1] y[2] x[2]
    qc1 = M.constraint("qc1", z1, mosek.fusion.Domain.inQCone())
    qc2 = M.constraint("qc2", z2, mosek.fusion.Domain.inRotatedQCone())

    M.objective("obj", mosek.fusion.ObjectiveSense.Minimize, mosek.fusion.Expr.sum(y))
    M.solve()
    # M.writeTask('cqo1.ptf')

    # Get the linear solution values
    solx = x.level() #[0.26092041 0.26092041 0.23907959]
    soly = y.level() #[0.36899718 0.1690548  0.1690548 ]
    qc1lvl = qc1.level() #[0.36899718 0.26092041 0.26092041]
    qc1sn = qc1.dual() #[ 1.         -0.70710678 -0.70710678]


# power cone optimization
# https://docs.mosek.com/latest/pythonfusion/tutorial-pow-shared.html
with mosek.fusion.Model('pow1') as M:
    x  = M.variable('x', 3)
    x3 = M.variable()
    x4 = M.variable()

    M.constraint(mosek.fusion.Expr.dot(x, [1.0, 1.0, 0.5]), mosek.fusion.Domain.equalsTo(2.0))
    # Create the power cone constraints
    M.constraint(mosek.fusion.Var.vstack(x.slice(0,2), x3), mosek.fusion.Domain.inPPowerCone(0.2))
    M.constraint(mosek.fusion.Expr.vstack(x.index(2), 1.0, x4), mosek.fusion.Domain.inPPowerCone(0.4))
    M.objective(mosek.fusion.ObjectiveSense.Maximize, mosek.fusion.Expr.dot([1.0,1.0,-1.0], mosek.fusion.Var.vstack(x3, x4, x.index(0))))
    M.solve()
    solx = x.level() #[0.06393849 0.78328038 2.30556229]


# conic exponential optimization
# https://docs.mosek.com/latest/pythonfusion/tutorial-ceo-shared.html
with mosek.fusion.Model('ceo1') as M:
    x = M.variable('x', 3)

    M.constraint("lc", mosek.fusion.Expr.sum(x), mosek.fusion.Domain.equalsTo(1))
    # Create the conic exponential constraint
    expc = M.constraint("expc", x, mosek.fusion.Domain.inPExpCone())
    M.objective("obj", mosek.fusion.ObjectiveSense.Minimize, mosek.fusion.Expr.sum(x.slice(0,2)))
    M.solve()
    solx = x.level() #[0.61178825 0.17040005 0.2178117 ]
    expcval = expc.level() #[0.61178825 0.17040005 0.2178117 ]
    expcdual = expc.dual() #[ 0.21781171  0.21781171 -0.7821883 ]


# geometric programming
# https://docs.mosek.com/latest/pythonfusion/tutorial-gp-shared.html
# Models log(sum(exp(Ax+b))) <= 0.
# Each row of [A b] describes one of the exp-terms
def logsumexp(M, A, x, b):
    k = int(A.shape[0])
    u = M.variable(k)
    M.constraint(mosek.fusion.Expr.sum(u), mosek.fusion.Domain.equalsTo(1.0))
    tmp0 = mosek.fusion.Expr.hstack(u, mosek.fusion.Expr.constTerm(k, 1.0),
            mosek.fusion.Expr.add(mosek.fusion.Expr.mul(A, x), b))
    M.constraint(tmp0, mosek.fusion.Domain.inPExpCone())

Aw, Af, alpha, beta, gamma, delta = 200.0, 50.0, 2.0, 10.0, 2.0, 10.0
with mosek.fusion.Model('max_vol_box') as M:
    xyz = M.variable(3)
    M.objective('Objective', mosek.fusion.ObjectiveSense.Maximize, mosek.fusion.Expr.sum(xyz))
    logsumexp(M, np.array([[1,1,0],[1,0,1]]), xyz, np.array([np.log(2.0/Aw), np.log(2.0/Aw)]))
    M.constraint(mosek.fusion.Expr.dot([0, 1, 1], xyz), mosek.fusion.Domain.lessThan(np.log(Af)))
    M.constraint(mosek.fusion.Expr.dot([1,-1, 0], xyz), mosek.fusion.Domain.inRange(np.log(alpha),np.log(beta)))
    M.constraint(mosek.fusion.Expr.dot([0,-1, 1], xyz), mosek.fusion.Domain.inRange(np.log(gamma),np.log(delta)))
    # M.setLogHandler(sys.stdout)
    M.solve()
    h,w,d = np.exp(xyz.level())
print("h={0:.3f}, w={1:.3f}, d={2:.3f}".format(h, w, d))


# semidefinite
