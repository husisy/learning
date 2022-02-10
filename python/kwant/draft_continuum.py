import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.sparse.linalg
plt.ion()

import kwant
import kwant.continuum

from utils import pauli

lat_graphene = kwant.lattice.honeycomb()

# Processing continuum Hamiltonians with discretize: tight-binding approximation of continuous Hamiltonians

## https://kwant-project.org/doc/1/tutorial/discretize#using-discretize-to-obtain-a-template
z0 = kwant.continuum.discretize('k_x * A(x) * k_x')
print(z0)


## https://kwant-project.org/doc/1/tutorial/discretize#building-a-kwant-system-from-the-template
def hf_shape_stadium(site):
    (x, y) = site.pos
    ret = max(abs(x)-20, 0)**2 + y**2 < 30**2
    return ret
dev_stadium = kwant.Builder()
template = kwant.continuum.discretize("k_x**2 + k_y**2 + V(x, y)")
dev_stadium.fill(template, hf_shape_stadium, (0,0))
dev_stadium_f = dev_stadium.finalized()
ind_mode = 2
hf_potential = lambda x,y: 0.0003*x + 0.0005*y
ham = dev_stadium_f.hamiltonian_submatrix(params={'V':hf_potential}, sparse=True)
EVC = scipy.sparse.linalg.eigsh(ham, k=10, which='SM')[1]
kwant.plotter.map(dev_stadium_f, np.abs(EVC[:,ind_mode])**2)


## Bernevig-Hughes-Zhang (BHZ) model
# https://kwant-project.org/doc/1/tutorial/discretize#models-with-more-structure-bernevig-hughes-zhang
def hf_shape_center(site):
    (x, y) = site.pos
    ret = (0 <= y < param_qsh['W']) and (0 <= x < param_qsh['L'])
    return ret
def hf_shape_lead(site):
    (x, y) = site.pos
    ret = (0 <= y < param_qsh['W'])
    return ret
param_qsh = {
    'a': 20,
    'L': 2000, #L
    'W': 1000, #W
}
param_qsh_H = {'A':3.65, 'B':-68.6, 'D':-51.1, 'M':-0.01, 'C':0}
hamiltonian = """
    + C * identity(4) + M * kron(sigma_0, sigma_z)
    - B * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z)
    - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
    + A * k_x * kron(sigma_z, sigma_x)
    - A * k_y * kron(sigma_0, sigma_y)
"""
template = kwant.continuum.discretize(hamiltonian, grid=param_qsh['a'])
syst = kwant.Builder()
syst.fill(template, hf_shape_center, (0,0))
lead = kwant.Builder(kwant.TranslationalSymmetry([-param_qsh['a'], 0]))
lead.fill(template, hf_shape_lead, (0,0))
syst.attach_lead(lead)
syst.attach_lead(lead.reversed())
syst_f = syst.finalized()

kx = np.linspace(-0.3, 0.3, 201)
fig,ax = plt.subplots()
kwant.plotter.bands(syst_f.leads[0], params=param_qsh_H, momenta=kx, ax=ax)
ax.grid()
ax.set_xlim(kx.min(), kx.max())
ax.set_ylim(-0.05, 0.05)
ax.set_xlabel('momentum [1/A]')
ax.set_ylabel('energy [eV]')

# get scattering wave functions at E=0
wf = kwant.wave_function(syst_f, energy=0, params=param_qsh_H)
prob_density = kwant.operator.Density(syst_f, np.eye(4))
spin_density = kwant.operator.Density(syst_f, np.kron(pauli.sz, np.eye(2)))
wf_sqr = sum(prob_density(psi) for psi in wf(0))  # states from left lead
rho_sz = sum(spin_density(psi) for psi in wf(0))  # states from left lead

fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(8, 5))
kwant.plotter.map(syst_f, wf_sqr, ax=ax1, cmap='OrRd')
im = [obj for obj in ax1.get_children() if isinstance(obj, matplotlib.image.AxesImage)][0]
fig.colorbar(im, ax=ax1)
kwant.plotter.map(syst_f, rho_sz, ax=ax2, cmap='coolwarm')
im = [obj for obj in ax2.get_children() if isinstance(obj, matplotlib.image.AxesImage)][0]
fig.colorbar(im, ax=ax2)
ax1.set_title('Probability density')
ax2.set_title('Spin density')


## lattice spacing https://kwant-project.org/doc/1/tutorial/discretize#limitations-of-discretization
def plot(ax, a, alpha):
    hamiltonian = "k_x**2 * identity(2) + alpha * k_x * sigma_y"
    h_k = kwant.continuum.lambdify(hamiltonian, locals={'alpha':alpha})
    k_cont = np.linspace(-4, 4, 201)
    e_cont = [scipy.linalg.eigvalsh(h_k(k_x=ki)) for ki in k_cont]

    template = kwant.continuum.discretize(hamiltonian, grid=a)
    syst = kwant.wraparound.wraparound(template).finalized()
    k_tb = np.linspace(-np.pi/a, np.pi/a, 201)
    e_tb = [scipy.linalg.eigvalsh(syst.hamiltonian_submatrix(params={'k_x':a*ki, 'alpha':alpha})) for ki in k_tb]

    ax.plot(k_cont, e_cont, 'r-')
    ax.plot(k_tb, e_tb, 'k-')
    ax.plot([], [], 'r-', label='continuum')
    ax.plot([], [], 'k-', label='tight-binding')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-1, 14)
    ax.set_title('a={}'.format(a))
    ax.set_xlabel('momentum [a.u.]')
    ax.set_ylabel('energy [a.u.]')
    ax.grid()
    ax.legend()
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(12, 4))
plot(ax0, a=1, alpha=0.5)
plot(ax1, a=0.25, alpha=0.5)


## https://kwant-project.org/doc/1/tutorial/discretize#advanced-topics
subs = {'sx': [[0, 1], [1, 0]], 'sz': [[1, 0], [0, -1]]}
e = (
    kwant.continuum.sympify('[[k_x**2, alpha * k_x], [k_x * alpha, -k_x**2]]'),
    kwant.continuum.sympify('k_x**2 * sigma_z + alpha * k_x * sigma_x'),
    kwant.continuum.sympify('k_x**2 * sz + alpha * k_x * sx', locals=subs),
)
print(e[0] == e[1] == e[2])

subs = {'A': 'A(x) + B', 'V': 'V(x) + V_0', 'C': 5}
print(kwant.continuum.sympify('k_x * A * k_x + V + C', locals=subs))
