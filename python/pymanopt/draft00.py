import numpy as np
import torch

import pymanopt

np_rng = np.random.default_rng()


def demo_manifold_sphere():
    dim = 3
    manifold = pymanopt.manifolds.Sphere(dim)
    x0 = manifold.random_point()
    assert (np.linalg.norm(x0)-1) < 1e-6


def demo_rayleigh_quotient_real():
    dim = 4
    tmp0 = np_rng.uniform(-1,1,size=(dim,dim))
    matA = tmp0 + tmp0.T

    manifold = pymanopt.manifolds.Sphere(dim)
    @pymanopt.function.autograd(manifold)
    def cost(point):
        return point @ matA @ point

    prob = pymanopt.Problem(manifold, cost)
    # optimizer = pymanopt.optimizers.SteepestDescent()
    optimizer = pymanopt.optimizers.TrustRegions()
    result = optimizer.run(prob)
    assert abs(np.linalg.norm(result.point) - 1) < 1e-6
    assert abs(result.cost - np.linalg.eigvalsh(matA)[0]) < 1e-6

def demo_rayleigh_quotient_complex():
    dim = 4
    tmp0 = np_rng.uniform(-1,1,size=(dim,dim)) + 1j*np_rng.uniform(-1,1,size=(dim,dim))
    matA = tmp0 + tmp0.T.conj()
    matA_torch = torch.tensor(matA, dtype=torch.complex128)

    manifold = pymanopt.manifolds.Sphere(dim*2)
    @pymanopt.function.pytorch(manifold)
    def cost(point):
        tmp0 = point[::2] + 1j*point[1::2]
        loss = (tmp0 @ matA_torch @ tmp0.conj()).real
        return loss

    prob = pymanopt.Problem(manifold, cost)
    # optimizer = pymanopt.optimizers.SteepestDescent()
    optimizer = pymanopt.optimizers.TrustRegions()
    result = optimizer.run(prob)
    assert abs(np.linalg.norm(result.point) - 1) < 1e-6
    assert abs(result.cost - np.linalg.eigvalsh(matA)[0]) < 1e-6
