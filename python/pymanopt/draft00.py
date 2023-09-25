import numpy as np

import pymanopt

np_rng = np.random.default_rng()

dim = 3
manifold = pymanopt.manifolds.Sphere(dim)

tmp0 = np_rng.normal(size=(dim,dim))
matrix = tmp0 + tmp0.T

@pymanopt.function.autograd(manifold)
def cost(point):
    return point @ matrix @ point

problem = pymanopt.Problem(manifold, cost)
optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)
assert abs(np.linalg.norm(result.point) - 1) < 1e-6
assert abs(result.cost - np.linalg.eigvalsh(matrix)[0]) < 1e-6
