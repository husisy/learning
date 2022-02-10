# https://kwant-project.org/doc/1/tutorial/
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
plt.ion()

import kwant

tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]

lat = kwant.lattice.square()


## Band structure calculations
# https://kwant-project.org/doc/1/tutorial/spectrum#band-structure-calculations
param_wire = {
    't': 1,
    'width': 10,
}
dev_wire = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
dev_wire[(lat(0,j) for j in range(param_wire['width']))] = 4*param_wire['t']
dev_wire[((lat(0,j),lat(0,j-1)) for j in range(1,param_wire['width']))] = -param_wire['t']
dev_wire[((lat(1,j),lat(0,j)) for j in range(param_wire['width']))] = -param_wire['t']
dev_wire_f = dev_wire.finalized()
dev_wire_kf = kwant.wraparound.wraparound(dev_wire).finalized()
# kwant.plotter.bands(dev_wire_f)
# kwant.physics.Bands
kx = np.linspace(-np.pi, np.pi, 101)
energy_band = np.stack([np.linalg.eigvalsh(dev_wire_kf.hamiltonian_submatrix(params={'k_x':x})) for x in kx], axis=0)
fig,ax = plt.subplots()
ax.plot(kx, energy_band, color=tableau_colorblind[1])
ax.set_xlim(kx.min(), kx.max())
ax.set_xlabel('kx')
ax.set_title('energy band')


## band structture of Closed systems
# https://kwant-project.org/doc/1/tutorial/spectrum#closed-systems
#  Fock-darwin spectrum of a quantum dot (energy spectrum in as a function of a magnetic field)
def make_ring(t, r):
    lat1 = kwant.lattice.square(norbs=1) #fail without norbs=1
    def hf_shape_circle(pos):
        (x, y) = pos
        ret = x**2+y**2 < r**2
        return ret
    def hopx(site1, site2, B):
        y = site1.pos[1]
        ret = -t * np.exp(-1j * B * y)
        return ret
    dev_ring = kwant.Builder()
    dev_ring[lat1.shape(hf_shape_circle, (0, 0))] = 4 * t
    dev_ring[kwant.builder.HoppingKind((1, 0), lat1, lat1)] = hopx #x-direction
    dev_ring[kwant.builder.HoppingKind((0, 1), lat1, lat1)] = -t #y-directions
    return dev_ring

param_ring = {
    't': 1,
    'r_small': 10,
    'r_large': 30,
    'B': 0.001,
}
Bfields = np.linspace(0, 0.2, 101)[:-1]
ind_mode = 9
dev_ring_small = make_ring(t=param_ring['t'], r=param_ring['r_small'])
dev_ring_small_f = dev_ring_small.finalized()
dev_ring_large = make_ring(t=param_ring['t'], r=param_ring['r_large'])
dev_ring_large_f = dev_ring_large.finalized()

# energy levels flow towards Landau level energies with increasing magnetic field.
energy_band = []
for B in Bfields:
    tmp0 = dev_ring_small_f.hamiltonian_submatrix(params={'B':B}, sparse=True)
    energy_band.append(scipy.sparse.linalg.eigsh(tmp0.tocsc(), k=15, sigma=0, return_eigenvectors=False)) #the lowest 15 eigenvalues
energy_band = np.stack(energy_band, axis=0)
fig,ax = plt.subplots()
ax.plot(Bfields, energy_band)
ax.set_xlabel('magnetic field')
ax.set_title('energy band')

ham_mat = dev_ring_large_f.hamiltonian_submatrix(sparse=True, params={'B':param_ring['B']})
EVL,EVC = scipy.sparse.linalg.eigsh(ham_mat.tocsc(), k=20, sigma=0)
ind0 = np.argsort(EVL)
EVL,EVC = EVL[ind0],EVC[:,ind0]

kwant.plotter.map(dev_ring_large_f, np.abs(EVC[:, ind_mode])**2, colorbar=False, oversampling=1) #plot wavefunction

#plot the local current
J = kwant.operator.Current(dev_ring_large_f)
current = J(EVC[:, ind_mode], params={'B':param_ring['B']}) #(np,float64,11000)
kwant.plotter.current(dev_ring_large_f, current, colorbar=False)
#TODO strange artifact, no error on the variable "current", it should be the bug in plotter
