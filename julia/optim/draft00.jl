using Optim

rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
function rosenbrock_grad!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

z0 = optimize(rosenbrock, zeros(2), BFGS())

z1 = optimize(rosenbrock, rosenbrock_grad!, zeros(2), BFGS())

optimize(rosenbrock, zeros(2), BFGS(); autodiff=:forward)
