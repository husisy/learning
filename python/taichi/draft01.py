import numpy as np

import taichi as ti

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))

ti.init(arch=ti.cpu)

ti0 = ti.var(dt=ti.f32, shape=(3,5))
np0 = np.random.randn(3, 5).astype(np.float32)
ti0.from_numpy(np0)
np1 = ti0.to_numpy()
# ti0.from_torch
# ti0.to_torch

z0 = ti.field(ti.float64, shape=(3,))
z1 = {'z0':0, 'z1':1, 'z2':2}
hf0 = lambda x: ti.static(z1[x])
@ti.kernel
def init0():
    for x in ti.static(range(3)):
        z0[hf0('z' + str(x))] = 233
@ti.kernel
def init2():
    z0[0] = ti.random(ti.f64)*np.pi
