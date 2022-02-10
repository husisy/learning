import numpy as np
import scipy.sparse

import matplotlib.pyplot as plt
plt.ion()

import kwant

lat_graphene = kwant.lattice.honeycomb(norbs=1)

def hf_check_real(np0, eps=1e-10):
    assert (-eps<np0.imag.min()) and (np0.imag.max() < eps)
    ret = np0.real.copy()
    return ret

def hf_dummy_real_KPM(np0, np1, eps=1e-10):
    np1 = hf_check_real(np1, eps)
    return np0, np1

def hf_Rademacher():
    while True:
        yield np.random.randint(2, size=shape0)*2 - 1

## Kernel Polynomial Method, spectral density, Chebyshev polynomials, random trace approximation
# https://kwant-project.org/doc/1/tutorial/kpm
def make_disk0(radius, t, m):
    def hf_shape_circle(pos):
        x, y = pos
        ret = x ** 2 + y ** 2 < radius ** 2
        return ret
    ret = kwant.Builder()
    ret[lat_graphene.a.shape(hf_shape_circle, (0, 0))] = m
    ret[lat_graphene.b.shape(hf_shape_circle, (0, 0))] = -m
    ret[lat_graphene.neighbors()] = t
    ret.eradicate_dangling()
    return ret


def make_Haldane_model(radius, t, t2):
    def hf_shape_circle(pos):
        x, y = pos
        ret = x**2 + y**2 < radius** 2
        return ret
    syst = kwant.Builder()
    syst[lat_graphene.shape(hf_shape_circle, (0, 0))] = 0.
    syst[lat_graphene.neighbors()] = t
    # second neighbours hoppings
    syst[lat_graphene.a.neighbors()] = 1j * t2
    syst[lat_graphene.b.neighbors()] = -1j * t2
    syst.eradicate_dangling()
    return syst


dev_disk0 = make_disk0(radius=30, t=-1, m=0)
dev_disk0_f = dev_disk0.finalized()

# density of state
spectrum = kwant.kpm.SpectralDensity(dev_disk0_f)
energy_full,density_full = hf_dummy_real_KPM(*spectrum())
energy_part = np.linspace(0, 2, 50)
density_part = hf_check_real(spectrum(energy_part))
fig,ax = plt.subplots()
ax.plot(energy_full, density_full, label='density')
ax.plot(energy_part, density_part, label='density_part')
ax.legend()
ax.set_xlabel("energy (t)")
ax.set_ylabel('DOS')

# integration using KPM
len(dev_disk0_f.sites) #6493
num_state = hf_check_real(spectrum.integrate()) #6493.000000000001
hf_fermi = lambda E: 1 / (np.exp((E-0.1)/0.2) + 1) #Fermi_energy=0.1, temperature=0.2
num_filled_state = hf_check_real(spectrum.integrate(hf_fermi)) #3289.6857658576496

# increase the accuracy of the approximation
spectrum = kwant.kpm.SpectralDensity(dev_disk0_f)
energy0,dos0 = hf_dummy_real_KPM(*spectrum()) #(np,float64,200)
spectrum.add_moments(energy_resolution=0.03)
energy1,dos1 = hf_dummy_real_KPM(*spectrum()) #(np,float64,526)
spectrum.add_moments(100)
spectrum.add_vectors(5)
energy2,dos2 = hf_dummy_real_KPM(*spectrum()) #(np,float64,726)
fig,ax = plt.subplots()
ax.plot(energy0, dos0, label='default', linewidth=1)
ax.plot(energy1, dos1, label='higher energy resolution', linewidth=1)
ax.plot(energy2, dos2, label='more moments and vectors', linewidth=1)
ax.legend()
ax.set_xlabel("energy (t)")
ax.set_ylabel('DOS')

# spectral density of an operator
op0 = scipy.sparse.eye(len(dev_disk0_f.sites))
spectrum_op0 = kwant.kpm.SpectralDensity(dev_disk0_f, operator=op0)
energy0,dos0 = hf_dummy_real_KPM(*spectrum_op0())
op1 = kwant.operator.Density(dev_disk0_f, sum=True) #sum over all the sites
spectrum_op1 = kwant.kpm.SpectralDensity(dev_disk0_f, operator=op1)
energy1,dos1 = hf_dummy_real_KPM(*spectrum_op1())
fig,ax = plt.subplots()
ax.plot(energy0, dos0, label='identity matrix')
ax.plot(energy1, dos1, label='kwant.operator.Density')
ax.legend()
ax.set_xlabel("energy (t)")
ax.set_ylabel('DOS')

# local density of state
ldos_op = kwant.operator.Density(dev_disk0_f, sum=False)
spectrum_ldos = kwant.kpm.SpectralDensity(dev_disk0_f, operator=ldos_op)
ldos_E0 = spectrum_ldos(energy=0)
ldos_E1 = spectrum_ldos(energy=1)
fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,5))
kwant.plotter.density(dev_disk0_f, ldos_E0.real, ax=ax0)
ax0.set_title('energy=0')
kwant.plotter.density(dev_disk0_f, ldos_E1.real, ax=ax1)
ax1.set_title('energy=1')
fig.tight_layout()

# vector_factory_example
spectrum0 = kwant.kpm.SpectralDensity(dev_disk0_f)
energy0,dos0 = hf_dummy_real_KPM(*spectrum0())
shape0 = dev_disk0_f.hamiltonian_submatrix(sparse=True).shape[0]
spectrum1 = kwant.kpm.SpectralDensity(dev_disk0_f, vector_factory=hf_Rademacher())
energy1,dos1 = hf_dummy_real_KPM(*spectrum1())
fig,ax = plt.subplots()
ax.plot(energy0, dos0, label='default')
ax.plot(energy1, dos1, label='Rademacher vector')
ax.legend()
ax.set_xlabel("energy (t)")
ax.set_ylabel('DOS')

# bilinear_map_operator_example
rho = kwant.operator.Density(dev_disk0_f, sum=True)
rho_spectrum = kwant.kpm.SpectralDensity(dev_disk0_f, operator=rho)
energy0,dos0 = hf_dummy_real_KPM(*rho_spectrum())
hf0 = lambda bra,ket: np.vdot(bra,ket) #sesquilinear map that does the same thing as `rho`
rho_alt_spectrum = kwant.kpm.SpectralDensity(dev_disk0_f, operator=hf0)
energy1,dos1 = hf_dummy_real_KPM(*rho_alt_spectrum())
fig,ax = plt.subplots()
ax.plot(energy0, dos0, label='kwant.operator.Density')
ax.plot(energy1, dos1, label='bilinear operator')
ax.legend()
ax.set_xlabel("energy (t)")
ax.set_ylabel('DOS')


## local spectral density using local vectors
dev_disk_staggered = make_disk0(radius=30, t=-1, m=0.1)
dev_disk_staggered_f = dev_disk_staggered.finalized()
# find 'A' and 'B' sites in the unit cell at the center of the disk
hf0 = lambda x: (x.tag[0]==0) and (x.tag[1]==0)
vector_factory = kwant.kpm.LocalVectors(dev_disk_staggered_f, where=hf0)
# 'num_vectors' can be unspecified when using 'LocalVectors'
spectrum_ldos = kwant.kpm.SpectralDensity(dev_disk_staggered_f, num_vectors=None, vector_factory=vector_factory, mean=False)
energy, tmp0 = spectrum_ldos()
densityA,densityB = tmp0.T
fig,ax = plt.subplots()
ax.plot(energy, densityA, label='sublattice A')
ax.plot(energy, densityB, label='sublattice B')
ax.legend()
ax.set_xlabel("energy (t)")
ax.set_ylabel('DOS')


## Haldane model
dev_Haldane = make_Haldane_model(radius=30, t=1, t2=0.5)
dev_Haldane_f = dev_Haldane.finalized()
area_per_site = np.abs(np.linalg.det(lat_graphene.prim_vecs)) / len(lat_graphene.sublattices)
hf_where = lambda s: np.linalg.norm(s.pos) < 1 #sites around the center unit cell of the disk
cond_xx = kwant.kpm.conductivity(dev_Haldane_f, alpha='x', beta='x', mean=True,
            num_vectors=None, vector_factory=kwant.kpm.LocalVectors(dev_Haldane_f, hf_where))
cond_xy = kwant.kpm.conductivity(dev_Haldane_f, alpha='x', beta='y', mean=True,
            num_vectors=None, vector_factory=kwant.kpm.LocalVectors(dev_Haldane_f, hf_where))
spectrum = kwant.kpm.SpectralDensity(dev_Haldane_f, num_vectors=None, vector_factory=kwant.kpm.LocalVectors(dev_Haldane_f, hf_where))
energy = cond_xx.energies
cond_xx = np.array([cond_xx(x, temperature=0.01) for x in energy]) / area_per_site
cond_xy = np.array([cond_xy(x, temperature=0.01) for x in energy]) / area_per_site
fig,ax = plt.subplots()
ax.fill_between(spectrum.energies, spectrum.densities*8, label="DoS [a.u.]", alpha=0.5, color='gray')
ax.plot(energy, cond_xx, label=r'$\sigma_{xx}$')
ax.plot(energy, cond_xy, label=r'$\sigma_{xy}$')
ax.legend()
ax.set_xlabel('energy ($t$)')
ax.set_ylabel(r'$\sigma [e^2/h]$')
