import taichi as ti
import numpy as np
import time

res = 600
dx = 1.0
half_inv_dx = 0.5 / dx
dt = 0.03
p_jacobi_iters = 30
f_strength = 10000.0
dye_decay = 0.99

ti.init(arch=ti.opengl)

_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
_new_velocities = ti.Vector(2, dt=ti.f32, shape=(res, res))
velocity_divs = ti.var(dt=ti.f32, shape=(res, res))
_pressures = ti.var(dt=ti.f32, shape=(res, res))
_new_pressures = ti.var(dt=ti.f32, shape=(res, res))
color_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))
_dye_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))
_new_dye_buffer = ti.Vector(3, dt=ti.f32, shape=(res, res))


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur


velocities_pair = TexPair(_velocities, _new_velocities)
pressures_pair = TexPair(_pressures, _new_pressures)
dyes_pair = TexPair(_dye_buffer, _new_dye_buffer)


@ti.func
def clip_index(array2d, ind0, ind1):
    N0,N1 = array2d.shape()
    ret = array2d[max(0, min(ind0, N0-1)), max(0, min(ind1, N1-1))]
    return ret


@ti.func
def bilinear_interp(vf, s, t):
    iu, iv = int(s), int(t)
    fu, fv = s - iu, t - iv
    a = clip_index(vf, iu, iv)
    b = clip_index(vf, iu+1, iv)
    c = clip_index(vf, iu, iv+1)
    d = clip_index(vf, iu+1, iv+1)
    tmp0 = a + fu*(b-a)
    tmp1 = c + fu*(d-c)
    ret = tmp0 + fv*(tmp1-tmp0)
    return ret


@ti.kernel
def advect(vf: ti.template(), qf: ti.template(), new_qf: ti.template()):
    for i, j in vf:
        tmp0 = i - dt*vf[i,j][0]
        tmp1 = j - dt*vf[i,j][1]
        new_qf[i, j] = bilinear_interp(qf, tmp0, tmp1)


def np_advect(vf, qf, dt=dt):
    dtype = vf.dtype
    N0,N1,_ = vf.shape
    s = np.arange(N0, dtype=dtype)[:,np.newaxis] - dt*vf[:,:,0]
    t = np.arange(N1, dtype=dtype) - dt*vf[:,:,1]
    fu,iu = np.modf(s)
    iu = iu.astype(np.int32)
    fv,iv = np.modf(t)
    iv = iv.astype(np.int32)
    ind0 = np.clip(iu, 0, N0-1)
    ind1 = np.clip(iu+1, 0, N0-1)
    ind2 = np.clip(iv, 0, N1-1)
    ind3 = np.clip(iv+1, 0, N1-1)
    a = qf[ind0, ind2]
    b = qf[ind1, ind2]
    c = qf[ind0, ind3]
    d = qf[ind1, ind3]
    tmp0 = a + fu[:,:,np.newaxis]*(b-a)
    tmp1 = c + fu[:,:,np.newaxis]*(d-c)
    ret = tmp0 + fv[:,:,np.newaxis]*(tmp1-tmp0)
    return ret


# from zzz import to_pickle, from_pickle, hfe
# zc0 = velocities_pair.cur
# zc1 = dyes_pair.cur
# zc2 = dyes_pair.nxt
# rand_state = np.random.RandomState(233)
# zc0.from_numpy(rand_state.rand(600,600,2).astype(np.float32))
# zc1.from_numpy(rand_state.rand(600,600,2).astype(np.float32))
# zc234 = np_advect(zc0.to_numpy(), zc1.to_numpy())
# advect(zc0, zc1, zc2)
# zc233 = zc2.to_numpy()
# print(hfe(zc233, zc234))
# print(np.abs(zc233-zc234).max())


force_radius = res / 3.0
inv_force_radius = 1.0 / force_radius
inv_dye_denom = 4.0 / (res / 15.0)**2
f_strength_dt = f_strength * dt


@ti.kernel
def apply_impulse(vf: ti.template(), dyef: ti.template(), imp_data: ti.ext_arr()):
    for i, j in vf:
        omx, omy = imp_data[2], imp_data[3]
        mdir = ti.Vector([imp_data[0], imp_data[1]])
        dx, dy = (i + 0.5 - omx), (j + 0.5 - omy)
        d2 = dx * dx + dy * dy
        # dv = F * dt
        factor = ti.exp(-d2 * inv_force_radius)
        momentum = mdir * f_strength_dt * factor
        v = vf[i, j]
        vf[i, j] = v + momentum
        # add dye
        dc = dyef[i, j]
        if mdir.norm() > 0.5:
            dc += ti.exp(-d2 * inv_dye_denom) * ti.Vector([imp_data[4], imp_data[5], imp_data[6]])
        dc *= dye_decay
        dyef[i, j] = dc


@ti.kernel
def divergence(vf: ti.template()):
    for i, j in vf:
        vl = clip_index(vf, i-1, j)[0]
        vr = clip_index(vf, i+1, j)[0]
        vb = clip_index(vf, i, j-1)[1]
        vt = clip_index(vf, i, j+1)[1]
        vc = clip_index(vf, i, j)
        if i == 0:
            vl = -vc[0]
        if i == res - 1:
            vr = -vc[0]
        if j == 0:
            vb = -vc[1]
        if j == res - 1:
            vt = -vc[1]
        velocity_divs[i, j] = (vr - vl + vt - vb) * half_inv_dx


p_alpha = -dx * dx


@ti.kernel
def pressure_jacobi(pf: ti.template(), new_pf: ti.template()):
    for i, j in pf:
        pl = clip_index(pf, i-1, j)
        pr = clip_index(pf, i+1, j)
        pb = clip_index(pf, i, j-1)
        pt = clip_index(pf, i, j+1)
        div = velocity_divs[i, j]
        new_pf[i, j] = (pl + pr + pb + pt + p_alpha * div) * 0.25


@ti.kernel
def subtract_gradient(vf: ti.template(), pf: ti.template()):
    for i, j in vf:
        pl = clip_index(pf, i-1, j)
        pr = clip_index(pf, i+1, j)
        pb = clip_index(pf, i, j-1)
        pt = clip_index(pf, i, j+1)
        v = clip_index(vf, i, j)
        v = v - half_inv_dx * ti.Vector([pr - pl, pt - pb])
        vf[i, j] = v


@ti.kernel
def fill_color_v2(vf: ti.template()):
    for i, j in vf:
        v = vf[i, j]
        color_buffer[i, j] = ti.Vector([abs(v[0]), abs(v[1]), 0.25])


@ti.kernel
def fill_color_v3(vf: ti.template()):
    for i, j in vf:
        v = vf[i, j]
        color_buffer[i, j] = ti.Vector([abs(v[0]), abs(v[1]), abs(v[2])])


@ti.kernel
def fill_color_s(sf: ti.template()):
    for i, j in sf:
        s = abs(sf[i, j])
        color_buffer[i, j] = ti.Vector([s, s * 0.25, 0.2])


def step(mouse_data):
    # tmp0 = np_advect(velocities_pair.cur.to_numpy(), velocities_pair.cur.to_numpy())
    # velocities_pair.nxt.from_numpy(tmp0)
    advect(velocities_pair.cur, velocities_pair.cur, velocities_pair.nxt)
    # tmp0 = np_advect(velocities_pair.cur.to_numpy(), dyes_pair.cur.to_numpy())
    # dyes_pair.nxt.from_numpy(tmp0)
    advect(velocities_pair.cur, dyes_pair.cur, dyes_pair.nxt)
    velocities_pair.swap()
    dyes_pair.swap()

    apply_impulse(velocities_pair.cur, dyes_pair.cur, mouse_data)

    divergence(velocities_pair.cur)
    for _ in range(p_jacobi_iters):
        pressure_jacobi(pressures_pair.cur, pressures_pair.nxt)
        pressures_pair.swap()

    subtract_gradient(velocities_pair.cur, pressures_pair.cur)
    fill_color_v3(dyes_pair.cur)
    # fill_color_s(velocity_divs)
    # fill_color_v2(velocities_pair.cur)


class MouseDataGen(object):
    def __init__(self):
        self.prev_mouse = None
        self.prev_color = None

    def __call__(self, gui):
        # [0:2]: normalized delta direction
        # [2:4]: current mouse xy
        # [4:7]: color
        if gui.is_pressed(ti.GUI.LMB):
            tmp0 = gui.get_cursor_pos() #(tuple,float,2) (x,y) in [0,1] from lower left corner
            # print('[info] key captured: ', type(tmp0), tmp0)
            mxy = np.array(tmp0, dtype=np.float32) * res
            if self.prev_mouse is None:
                self.prev_mouse = mxy
                self.prev_color = np.random.uniform(0.3, 0.7, size=3) #prevent too dark color
                ret = np.zeros(8, dtype=np.float32)
            else:
                mdir = mxy - self.prev_mouse
                mdir = mdir / (np.linalg.norm(mdir) + 1e-5)
                self.prev_mouse = mxy
                ret = np.array([*mdir, *mxy, *self.prev_color, 0], dtype=np.float32)
        else:
            # print('[info] no key captured')
            self.prev_mouse = None
            self.prev_color = None
            ret = np.zeros(8, dtype=np.float32)
        return ret


gui = ti.GUI('Stable-Fluid', (res, res))
md_gen = MouseDataGen()
while gui.running:
    gui.get_event(ti.GUI.PRESS) #DO nothing, just to capture key press

    mouse_data = md_gen(gui)
    step(mouse_data)

    img = color_buffer.to_numpy()
    gui.set_image(img)
    gui.show()
