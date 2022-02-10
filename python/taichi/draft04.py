# https://github.com/taichi-dev/taichi/blob/master/examples/simulation/cg_possion.py
import math
import taichi as ti

ti.init()

n = 512
steps = 32
eps = 1e-5

b = ti.field(float, (n, n))
x = ti.field(float, (n, n))
d = ti.field(float, (n, n))
r = ti.field(float, (n, n))


@ti.func
def c(x: ti.template(), i, j):
    ret = x[i,j] if (0<=i<n) and (0<=j<n) else 0.0
    return ret

@ti.func
def A(x: ti.template(), I):
    i, j = I
    ret = x[i,j]*4 - c(x,i-1,j) - c(x,i+1,j) - c(x,i,j-1) - c(x,i,j+1)
    return ret

@ti.kernel
def init():
    for I in ti.grouped(x):
        d[I] = b[I] - A(x, I)
        r[I] = d[I]

@ti.kernel
def substep():
    alpha, beta, dAd = 0.0, 0.0, eps
    for I in ti.grouped(x):
        dAd += d[I] * A(d, I)
    for I in ti.grouped(x):
        alpha += r[I]**2 / dAd
    for I in ti.grouped(x):
        x[I] = x[I] + alpha * d[I]
        r[I] = r[I] - alpha * A(d, I)
        beta += r[I]**2 / ((alpha + eps) * dAd)
    for I in ti.grouped(x):
        d[I] = r[I] + beta * d[I]


gui = ti.GUI('Possion Solver', (n, n))
while gui.running:
    for e in gui.get_events(gui.PRESS):
        if e.key == gui.ESCAPE:
            gui.running = False
        elif e.key == gui.LMB: #left mouse button
            b[int(e.pos[0]*n), int(e.pos[1]*n)] += 0.75
            init()
    for i in range(steps):
        substep()
    gui.set_image(x)
    gui.show()
