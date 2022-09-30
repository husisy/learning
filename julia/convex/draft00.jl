using Convex, SCS


m = 4
n = 5
A = randn(m, n)
b = randn(m, 1)
x = Variable(n)
object = sumsquares(A * x - b)
constraint = [x >= 0]
problem = minimize(object, constraint)
solve!(problem, SCS.Optimizer; silent_solver=true)
problem.status # :Optimal, :Infeasible, :Unbounded etc.
problem.optval #optimum value
evaluate(object)
