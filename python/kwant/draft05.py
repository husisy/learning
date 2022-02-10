# https://kwant-project.org/doc/1/tutorial/
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import kwant

from utils import pauli
tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]

lat2 = kwant.lattice.square(norbs=2)

## N-S interface, conservation laws, symmetry, Andreev reflection, superconducting gap
# https://kwant-project.org/doc/1/tutorial/superconductors
def check_particle_hole_symmetry(dev_sc_f):
    s = kwant.smatrix(dev_sc_f, energy=0)
    s_ee = s.submatrix((0,0), (0,0)) #electron to electron block
    s_hh = s.submatrix((0,1), (0,1)) #hole to hole block
    s_he = s.submatrix((0,1), (0,0)) #electron to hole block
    s_eh = s.submatrix((0,0), (0,1)) #hole to electron block
    print('s_ee:', np.round(s_ee, 3), sep='\n')
    print('s_hh:', np.round(s_hh[::-1, ::-1], 3), sep='\n')
    print('s_ee - s_hh^*:', np.round(s_ee - s_hh[::-1, ::-1].conj(), 3), sep='\n')
    print('s_he:', np.round(s_he, 3), sep='\n')
    print('s_eh:', np.round(s_eh[::-1, ::-1], 3), sep='\n')
    print('s_he + s_eh^*:', np.round(s_he + s_eh[::-1, ::-1].conj(), 3), sep='\n')
param_sc = {
    'width': 10,
    'length': 10,
    'barrier': 1.5,
    'barrierpos': (3,4),
    'mu': 0.4,
    'Delta': 0.1,
    'Deltapos': 4,
    't': 1,
}
dev_sc = kwant.Builder()
# onsite
tmp0 = (lat2(x, y) for x in range(param_sc['Deltapos']) for y in range(param_sc['width']))
dev_sc[tmp0] = (4*param_sc['t'] - param_sc['mu']) * pauli.tau_z
tmp0 = (lat2(x, y) for x in range(param_sc['Deltapos'], param_sc['length']) for y in range(param_sc['width']))
dev_sc[tmp0] = (4*param_sc['t'] - param_sc['mu']) * pauli.tau_z + param_sc['Delta'] * pauli.tau_x
tmp0 = (lat2(x, y) for x in range(param_sc['barrierpos'][0], param_sc['barrierpos'][1]) for y in range(param_sc['width']))
dev_sc[tmp0] = (4 * param_sc['t'] + param_sc['barrier'] - param_sc['mu']) * pauli.tau_z
# hopping
dev_sc[lat2.neighbors()] = -param_sc['t'] * pauli.tau_z

# Left lead - normal, so the order parameter is zero.
# Specify the conservation law used to treat electrons and holes separately.
# We only do this in the left lead, where the pairing is zero.
lead0 = kwant.Builder(kwant.TranslationalSymmetry((-1,0)), conservation_law=-pauli.tau_z, particle_hole=pauli.tau_y)
lead0[(lat2(0, j) for j in range(param_sc['width']))] = (4 * param_sc['t'] - param_sc['mu']) * pauli.tau_z
lead0[lat2.neighbors()] = -param_sc['t'] * pauli.tau_z
# Right lead - superconducting, so the order parameter is included.
lead1 = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
lead1[(lat2(0, j) for j in range(param_sc['width']))] = (4 * param_sc['t'] - param_sc['mu']) * pauli.tau_z + param_sc['Delta'] * pauli.tau_x
lead1[lat2.neighbors()] = -param_sc['t'] * pauli.tau_z
dev_sc.attach_lead(lead0)
dev_sc.attach_lead(lead1)
dev_sc_f = dev_sc.finalized()

check_particle_hole_symmetry(dev_sc_f)

# conductance=N - R_ee + R_he
energy = np.linspace(-0.02, 0.2, 100)
# target=(lead=0,block=1),source=(lead=0,block=0); block=0 means electron, block=1 means hole
hf0 = lambda x: (x.submatrix((0,0),(0,0)).shape[0], x.transmission((0,0),(0,0)), x.transmission((0,1),(0,0)))
tmp0 = np.array([hf0(kwant.smatrix(dev_sc_f, x)) for x in energy])
conductance_channel = tmp0[:,0]
conductance_ee = tmp0[:,1]
conductance_he = tmp0[:,2]
conductance = conductance_channel - conductance_ee + conductance_he
# conductance = np.array([hf0(kwant.smatrix(dev_sc_f, x)) for x in energy])
fig,ax = plt.subplots()
ax.plot(energy, conductance_channel, label='#channel')
ax.plot(energy, conductance_ee, label='$G_{ee}$')
ax.plot(energy, conductance_he, label='$G_{he}$')
ax.plot(energy, conductance, label='conductance')
ax.set_xlabel('energy (t)')
ax.set_ylabel('conductance ($e^2/h$)')
ax.legend()
