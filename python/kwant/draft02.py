# https://kwant-project.org/doc/1/tutorial/
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import kwant

from utils import pauli
tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]

lat = kwant.lattice.square()


## https://kwant-project.org/doc/1/tutorial/first_steps
param_wire = {
    't':1,
    'width': 10,
    'length': 30,
}
param_wire['length']
dev_wire = kwant.Builder()
dev_wire[(lat(x,y) for x in range(param_wire['length']) for y in range(param_wire['width']))] = 4*param_wire['t']
dev_wire[lat.neighbors()] = -param_wire['t']
lead_left = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
lead_left[(lat(0, y) for y in range(param_wire['width']))] = 4 * param_wire['t']
lead_left[lat.neighbors()] = -param_wire['t']
lead_right = kwant.Builder(kwant.TranslationalSymmetry([1, 0]))
lead_right[(lat(0, y) for y in range(param_wire['width']))] = 4 * param_wire['t']
lead_right[lat.neighbors()] = -param_wire['t']
# lead_right = lead_right.reversed()
dev_wire.attach_lead(lead_left)
dev_wire.attach_lead(lead_right)
# kwant.plot(dev_wire)
dev_wire_f = dev_wire.finalized()

energy = np.linspace(0, 1, 100)
# from lead0 to lead1
transmission = np.array([kwant.smatrix(dev_wire_f, x).transmission(1,0) for x in energy])

fig,ax = plt.subplots()
ax.plot(energy, transmission)
ax.set_xlabel('energy (t)')
ax.set_ylabel('conductance ($e^2/h$)')


## spin, square lattice, Rashba SOI, Zeeman spliting
# https://kwant-project.org/doc/1/tutorial/spin_potential_shape#matrix-structure-of-on-site-and-hopping-elements
# Physics background:
#  Gaps in quantum wires with spin-orbit coupling and Zeeman splititng as theoretically predicted in
#   http://prl.aps.org/abstract/PRL/v90/i25/e256601
#  and (supposedly) experimentally oberved in
#   http://www.nature.com/nphys/journal/v6/n5/abs/nphys1626.html
param_wire1 = {
    't': 1,
    'alpha': -0.5, #Rashba SOI
    'e_z': 0.01, #Zeeman
    'width': 10,
    'length': 30,
}
dev_wire1 = kwant.Builder()
dev_wire1[(lat(x, y) for x in range(param_wire1['length']) for y in range(param_wire1['width']))] = 4*param_wire1['t']*pauli.s0 + param_wire1['e_z']*pauli.sz
dev_wire1[kwant.builder.HoppingKind((1, 0), lat, lat)] = -param_wire1['t']*pauli.s0 + 1j*param_wire1['alpha']*pauli.sy/2 #x-direction
dev_wire1[kwant.builder.HoppingKind((0, 1), lat, lat)] = -param_wire1['t']*pauli.s0 - 1j*param_wire1['alpha']*pauli.sx/2 #y-directions
lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
lead[(lat(0, j) for j in range(param_wire1['width']))] = 4*param_wire1['t']*pauli.s0 + param_wire1['e_z']*pauli.sz
lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = -param_wire1['t']*pauli.s0 + 1j*param_wire1['alpha']*pauli.sy/2 #x-direction
lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = -param_wire1['t']*pauli.s0 - 1j*param_wire1['alpha']*pauli.sx/2 #y-directions
dev_wire1.attach_lead(lead)
dev_wire1.attach_lead(lead.reversed())
dev_wire1_f = dev_wire1.finalized()
lead_f = kwant.wraparound.wraparound(lead).finalized()
# kwant.plot(dev_wire1)

energy = np.linspace(-0.1, 0.4, 101)
transmission = np.array([kwant.smatrix(dev_wire1_f, x).transmission(1, 0) for x in energy])
fig,ax = plt.subplots()
ax.plot(energy, transmission)
ax.set_xlabel('energy (t)')
ax.set_ylabel('conductance ($e^2/h$)')

kx = np.linspace(-np.pi, np.pi, 101)
energy_band = np.stack([np.linalg.eigvalsh(lead_f.hamiltonian_submatrix(params={'k_x':x})) for x in kx], axis=0)
fig,ax = plt.subplots()
ax.plot(kx, energy_band, color=tableau_colorblind[1])
ax.set_ylim(energy.min(), 1)
ax.set_xlim(kx.min(), kx.max())


## transmission through a quantum well, Functions as values in Builder
# https://kwant-project.org/doc/1/tutorial/spin_potential_shape#spatially-dependent-values-through-functions
param_well = {
    't': 1,
    'width': 10, #W
    'length': 30, #L
    'well_length': 10, #L_well
}
def onsite(site, pot):
    (x, y) = site.pos
    tmp0 = ((param_well['length']-param_well['well_length'])/2<x) and (x<(param_well['length']+param_well['well_length'])/2)
    ret = 4 * param_well['t'] + (pot if tmp0 else 0)
    return ret
dev_well = kwant.Builder()
dev_well[(lat(x, y) for x in range(param_well['length']) for y in range(param_well['width']))] = onsite
dev_well[lat.neighbors()] = -param_well['t']
lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
lead[(lat(0, j) for j in range(param_well['width']))] = 4 * param_well['t']
lead[lat.neighbors()] = -param_well['t']
dev_well.attach_lead(lead)
dev_well.attach_lead(lead.reversed())
dev_well_f = dev_well.finalized()

welldepths = np.linspace(-1, 0, 101)
energy = 0.2
transmission = np.array([kwant.smatrix(dev_well_f, energy, params={'pot':x}).transmission(1, 0) for x in welldepths])
fig,ax = plt.subplots()
ax.plot(welldepths, transmission)
ax.set_xlabel('well depth (t)')
ax.set_ylabel('conductance ($e^2/h$)')



## Flux-dependent transmission through a quantum ring
# https://kwant-project.org/doc/1/tutorial/spin_potential_shape#nontrivial-shapes
def hf_shape_ring(pos):
    (x, y) = pos
    rsq = x ** 2 + y ** 2
    ret = (param_ring['r1']**2<rsq) and (rsq<param_ring['r2']**2)
    return ret
# Modify only those hopings in x-direction that cross the branch cut
def hf_shape_hops_across_cut(syst):
    ret = []
    for hop in kwant.builder.HoppingKind((1, 0), lat, lat)(syst):
        ix0, iy0 = hop[0].tag #same as hop[0].pos
        # builder.HoppingKind with the argument (1, 0) below returns hoppings ordered as ((i+1, j), (i, j))
        if (iy0 < 0) and (ix0 == 1): # ix1 == 0 then implied
            ret.append(hop)
    return ret
# In order to introduce a flux through the ring, we introduce a phase on the hoppings on the line cut through one of the arms
def hopping_phase(site1, site2, phi):
    ret = -param_ring['t']*np.exp(1j * phi)
    return ret
def hf_shape_lead(pos):
    (x, y) = pos
    ret = (-param_ring['width']/2<y) and (y<param_ring['width']/2)
    return ret
param_ring = {
    't': 1,
    'width': 10,
    'r1': 10,
    'r2': 20,
}
dev_ring = kwant.Builder()
dev_ring[lat.shape(hf_shape_ring, (0, param_ring['r1'] + 1))] = 4 * param_ring['t']
dev_ring[lat.neighbors()] = -param_ring['t']
dev_ring[hf_shape_hops_across_cut] = hopping_phase
lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
lead[lat.shape(hf_shape_lead, (0, 0))] = 4 * param_ring['t']
lead[lat.neighbors()] = -param_ring['t']
dev_ring.attach_lead(lead)
dev_ring.attach_lead(lead.reversed())
dev_ring_f = dev_ring.finalized()
# kwant.plot(dev_ring)

energy = 0.15
fluxes = np.linspace(0, 3*2*np.pi, 100)
transmission = np.array([kwant.smatrix(dev_ring_f, energy, params={'phi':x}).transmission(1, 0) for x in fluxes])
fig,ax = plt.subplots()
ax.plot(fluxes/(2*np.pi), transmission)
ax.set_xlabel('flux $h/e$')
ax.set_ylabel('conductance ($e^2/h$)')
