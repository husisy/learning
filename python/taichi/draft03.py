# https://github.com/taichi-dev/taichi/blob/master/examples/simulation/ad_gravity.py
import numpy as np
import taichi as ti

ti.init()

N = 8
dt = 1e-5
np_rng = np.random.default_rng()

x = ti.Vector.field(2, float, N, needs_grad=True)  # position of particles
v = ti.Vector.field(2, float, N)  # velocity of particles
U = ti.field(float, (), needs_grad=True)  # potential energy

@ti.kernel
def compute_U():
    for i, j in ti.ndrange(N, N):
        r = x[i] - x[j]
        # r.norm(1e-3) is equivalent to ti.sqrt(r.norm()**2 + 1e-3)
        U[None] += -1 / r.norm(1e-3)

@ti.kernel
def advance():
    for i in x:
        v[i] += dt * -x.grad[i]  # dv/dt = -dU/dx
    for i in x:
        x[i] += dt * v[i]

x.from_numpy(np_rng.uniform(0, 1, size=(N,2)))
gui = ti.GUI('Autodiff gravity')
while gui.running:
    for i in range(50):
        with ti.Tape(U):
            compute_U()
        advance()

    gui.circles(x.to_numpy(), radius=3)
    gui.show()
