import casadi

x0 = casadi.MX.sym('x0')
x1 = casadi.sin(x0)
casadi.jacobian(x1, x0)

x0 = casadi.SX.sym('x0')
x1 = casadi.sqrt(x0**2 + 10)

# constant
x0 = casadi.SX.zeros(2,3) #dense zeros
x1 = casadi.SX(2,3) #sparse zeros
x2 = casadi.SX.eye(3) #sparse eye

x0 = casadi.SX.sym('x0', 5) #list of symbols
x1 = casadi.SX.sym('x1', 4, 2)
