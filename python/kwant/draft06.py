import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import kwant

## spin texture (Skyrmions), local quantities
# https://kwant-project.org/doc/1/tutorial/operators

from utils import pauli

lat2 = kwant.lattice.square(norbs=2)

sigma_vec = np.stack([pauli.sx,pauli.sy,pauli.sz], axis=1)

def field_direction(pos, r0, delta):
    x, y = pos
    r = np.linalg.norm(pos)
    theta = (np.tanh((r-r0)/delta) - 1) * (np.pi / 2)
    if r == 0:
        m_i = np.array([0, 0, -1])
    else:
        m_i = np.array([(x/r)*np.sin(theta), (y/r)*np.sin(theta), np.cos(theta)])
    return m_i

def following_m_i(site, r0, delta):
    m_i = field_direction(site.pos, r0, delta)
    ret = np.dot(m_i, sigma_vec)
    return ret

def plot_vector_field(syst, params):
    xmin, ymin = min(s.tag for s in syst.sites)
    xmax, ymax = max(s.tag for s in syst.sites)
    x, y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    m_i = np.stack([field_direction(p, **params) for p in zip(x.flat, y.flat)], axis=0).reshape(x.shape + (3,)).transpose(2,0,1)
    fig, ax = plt.subplots(1, 1)
    im = ax.quiver(x, y, *m_i, pivot='mid', scale=75)
    fig.colorbar(im)

def plot_densities(syst, densities):
    fig, axes = plt.subplots(1, len(densities))
    if len(densities)==1:
        axes = (axes,)
    for ax, (title, rho) in zip(axes, densities):
        kwant.plotter.map(syst, rho, ax=ax, a=4)
        ax.set_title(title)

def plot_currents(syst, currents):
    fig, axes = plt.subplots(1, len(currents))
    if len(currents)==1:
        axes = (axes,)
    for ax, (title, current) in zip(axes, currents):
        kwant.plotter.current(syst, current, ax=ax, colorbar=False)
        ax.set_title(title)

def scattering_onsite(site, r0, delta, J):
    m_i = field_direction(site.pos, r0, delta)
    ret = J * np.dot(m_i, sigma_vec)
    return ret
def lead_onsite(site, J):
    ret = J * pauli.sz
    return ret
def hf_shape_square(pos):
    x,y = pos
    ret = (-param_wire['length']/2<x) and (x<param_wire['length']/2) and (-param_wire['length']/2<y) and (y<param_wire['length']/2)
    return ret
param_wire = {
    'length': 80,
}
dev_wire = kwant.Builder()
dev_wire[lat2.shape(hf_shape_square, (0, 0))] = scattering_onsite
dev_wire[lat2.neighbors()] = -pauli.s0
lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)), conservation_law=-pauli.sz)
lead[lat2.shape(hf_shape_square, (0, 0))] = lead_onsite
lead[lat2.neighbors()] = -pauli.s0
dev_wire.attach_lead(lead)
dev_wire.attach_lead(lead.reversed())
dev_wire_f = dev_wire.finalized()


plot_vector_field(dev_wire_f, {'r0':20, 'delta':10})

# calculate the expectation values of the operators with 'psi'
wf = kwant.wave_function(dev_wire_f, energy=-1, params={'r0':20, 'delta':10, 'J':1})
psi = wf(0)[0] #wf(lead=0)[state=0] the first scattering state from the left lead (np,complex128,12482)
# up, down = psi[::2], psi[1::2]
# density = np.abs(up)**2 + np.abs(down)**2
# spin_z = np.abs(up)**2 - np.abs(down)**2
# spin_y = 1j * (down.conjugate() * up - up.conjugate() * down)
density = kwant.operator.Density(dev_wire_f)(psi) #(np,float64,6241)
spin_z = kwant.operator.Density(dev_wire_f, pauli.sz)(psi)
spin_y = kwant.operator.Density(dev_wire_f, pauli.sy)(psi)
plot_densities(dev_wire_f, [
    (r'$\sigma_0$', density),
    (r'$\sigma_z$', spin_z),
    (r'$\sigma_y$', spin_y),
])


# calculate the expectation values of the operators with 'psi'
current = kwant.operator.Current(dev_wire_f)(psi) #(np,float64,24648)
spin_z_current = kwant.operator.Current(dev_wire_f, pauli.sz)(psi)
spin_y_current = kwant.operator.Current(dev_wire_f, pauli.sy)(psi)
plot_currents(dev_wire_f, [
    (r'$J_{\sigma_0}$', current),
    (r'$J_{\sigma_z}$', spin_z_current),
    (r'$J_{\sigma_y}$', spin_y_current),
])


m_current = kwant.operator.Current(dev_wire_f, following_m_i)(psi, params={'r0':25, 'delta':10})
plot_currents(dev_wire_f, [
    (r'$J_{\mathbf{m}_i}$', m_current),
    (r'$J_{\sigma_z}$', spin_z_current),
])



hf_shape_circle = lambda site: (np.linalg.norm(site.pos) < 20)
rho_circle = kwant.operator.Density(dev_wire_f, where=hf_shape_circle, sum=True)
all_states = np.concatenate([wf(0), wf(1)], axis=0)
dos_in_circle = sum(rho_circle(p) for p in all_states) / (2 * np.pi)
print('density of states in circle:', dos_in_circle)

def left_cut(site_to, site_from):
    return site_from.pos[0] <= -39 and site_to.pos[0] > -39

def right_cut(site_to, site_from):
    return site_from.pos[0] < 39 and site_to.pos[0] >= 39

J_left = kwant.operator.Current(dev_wire_f, where=left_cut, sum=True)
J_right = kwant.operator.Current(dev_wire_f, where=right_cut, sum=True)
Jz_left = kwant.operator.Current(dev_wire_f, pauli.sz, where=left_cut, sum=True)
Jz_right = kwant.operator.Current(dev_wire_f, pauli.sz, where=right_cut, sum=True)

print('J_left:', J_left(psi), ' J_right:', J_right(psi))
print('Jz_left:', Jz_left(psi), ' Jz_right:', Jz_right(psi))



## advanced topic
J_m = kwant.operator.Current(dev_wire_f, following_m_i)
J_z = kwant.operator.Current(dev_wire_f, pauli.sz)
J_m_bound = J_m.bind(params=dict(r0=25, delta=10, J=1))
J_z_bound = J_z.bind(params=dict(r0=25, delta=10, J=1))

# Sum current local from all scattering states on the left at energy=-1
wf_left = wf(0)
J_m_left = sum(J_m_bound(p) for p in wf_left)
J_z_left = sum(J_z_bound(p) for p in wf_left)
plot_currents(dev_wire_f, [
    (r'$J_{\mathbf{m}_i}$ (from left)', J_m_left),
    (r'$J_{\sigma_z}$ (from left)', J_z_left),
])
