import random
import numpy as np
import oommfc
import discretisedfield
import micromagneticmodel
import matplotlib.pyplot as plt

# %matplotlib inline

md = oommfc.MinDriver()
td = oommfc.TimeDriver()


## Current induced domain wall motion (Zhang-Li spin transfer torque)
##   https://oommfc.readthedocs.io/en/latest/ipynb/04-getting-started-current-induced-dw-motion.html
system = micromagneticmodel.System(name='domain_wall_pair')
system.energy = (micromagneticmodel.Exchange(A=15e-12)
            + micromagneticmodel.DMI(D=3e-3, crystalclass="Cnv")
            + micromagneticmodel.UniaxialAnisotropy(K=0.5e6, u=(0,0,1)))
system.dynamics = micromagneticmodel.Precession(gamma0=2.211e5) + micromagneticmodel.Damping(alpha=0.3)

L = 500e-9  # sample length (m)
w = 20e-9  # sample width (m)
d = 2.5e-9  # discretisation cell size (m)
region = discretisedfield.Region(p1=(0,0,0), p2=(L,w,d))
mesh = discretisedfield.Mesh(region=region, cell=(d,d,d))
def m_value(pos):
    x, y, z = pos
    ret = (0,0,-1) if (20e-9 < x < 40e-9) else (0, 0, 1)
    return ret
system.m = discretisedfield.Field(mesh, dim=3, value=m_value, norm=5.8e5)
md.drive(system)
# system.m.z.plane('z').k3d_scalar()

ux = 400  # velocity in x-direction (m/s)
beta = 0.5  # non-adiabatic STT parameter
system.dynamics += micromagneticmodel.ZhangLi(u=ux, beta=beta)
td.drive(system, t=0.5e-9, n=100)


## domain wallpair conversion, Zhang-Li spin transfer torque
##   https://oommfc.readthedocs.io/en/latest/ipynb/05-getting-started-exercise-dw-pair-conversion.html
system = micromagneticmodel.System(name='dw_pair_conversion')
system.energy = (micromagneticmodel.Exchange(A=15e-12)
            + micromagneticmodel.DMI(D=3e-3, crystalclass="Cnv")
            + micromagneticmodel.UniaxialAnisotropy(K=0.5e6, u=(0,0,1)))
system.dynamics = micromagneticmodel.Precession(gamma0=2.211e5) + micromagneticmodel.Damping(alpha=0.3)

region = discretisedfield.Region(p1=(0,0,0), p2=(150e-9,50e-9,2e-9))
mesh = discretisedfield.Mesh(region=region, cell=(2e-9,2e-9,2e-9))
def Ms_fun(pos):
    x, y, z = pos
    ret = 0 if (x < 50e-9) and (y < 15e-9 or y > 35e-9) else 5.8e5
    return ret
def m_init(pos):
    x, y, z = pos
    ret = (0.1, 0.1, -1) if (30e-9 < x < 40e-9) else (0.1, 0.1, 1)
    return ret
system.m = discretisedfield.Field(mesh, dim=3, value=m_init, norm=Ms_fun)
md.drive(system)
# system.m.z.plane('z').k3d_scalar(filter_field=system.m.norm)

system.dynamics += micromagneticmodel.ZhangLi(u=400, beta=0.5)
td.drive(system, t=0.2e-9, n=200)


## vortex dynamics
##   https://oommfc.readthedocs.io/en/latest/ipynb/06-getting-started-exercise-vortex-dynamics.html
region = discretisedfield.Region(p1=(-50e-9,-50e-9,0), p2=(50e-9, 50e-9, 5e-9))
mesh = discretisedfield.Mesh(region=region, cell=(5e-9, 5e-9, 5e-9))
def Ms_fun(pos):
    x, y, z = pos
    ret = 8e5 if ((x**2 + y**2)**0.5 < 50e-9) else 0
    return ret
def m_init(pos):
    x, y, z = pos
    A = 1e9  # (1/m)
    return -A*y, A*x, 10
system = micromagneticmodel.System(name='vortex_dynamics')
system.energy = micromagneticmodel.Exchange(A=13e-12) + micromagneticmodel.Demag()
system.dynamics = micromagneticmodel.Precession(gamma0=2.211e5) + micromagneticmodel.Damping(alpha=0.2)
system.m = discretisedfield.Field(mesh, dim=3, value=m_init, norm=Ms_fun)
md.drive(system)
# system.m.k3d_vector(color_field=system.m.z, head_size=10)

# vortex center is drifted towards negative y-axis
system.energy += micromagneticmodel.Zeeman(H=(1e4, 0, 0))
md.drive(system)
# system.m.k3d_vector(color_field=system.m.z, head_size=10)

system.energy.zeeman.H = (0, 0, 0)
td.drive(system, t=5e-9, n=500)
# system.table.data.plot('t', ['mx', 'my', 'mz'])


## spatially varying H

## negative exchange

## cubic anisotropy

## RKKY https://oommfc.readthedocs.io/en/latest/ipynb/rkky.html
region = discretisedfield.Region(p1=(0,0,0), p2=(60e-9, 60e-9, 22e-9))
subregions={'bottom': discretisedfield.Region(p1=(0, 0, 0), p2=(60e-9, 60e-9, 10e-9)),
            'spacer': discretisedfield.Region(p1=(0, 0, 10e-9), p2=(60e-9, 60e-9, 12e-9)),
            'top': discretisedfield.Region(p1=(0, 0, 12e-9), p2=(60e-9, 60e-9, 22e-9))}
mesh = discretisedfield.Mesh(region, n=(20, 20, 11), subregions=subregions)

# mesh.k3d_subregions()

system = micromagneticmodel.System(name='rkky')
system.energy = (micromagneticmodel.Exchange(A=1e-12)
            + micromagneticmodel.RKKY(sigma=-1e-4, sigma2=0, subregions=['bottom', 'top'])
            + micromagneticmodel.UniaxialAnisotropy(K=1e5, u=(1, 0, 0)))

def m_init(pos):
    return [2*random.random()-1 for i in range(3)]
norm = {'bottom': 8e6, 'top': 8e6, 'spacer': 0}
system.m = discretisedfield.Field(mesh, dim=3, value=m_init, norm=norm)
# system.m.plane('y').mpl(figsize=(15, 4))
md.drive(system)


## flower state (splayed state) https://oommfc.readthedocs.io/en/latest/ipynb/07-tutorial-standard-problem3.html
def m_init_flower(pos): #flower state
    x, y, z = pos[0]/1e-9, pos[1]/1e-9, pos[2]/1e-9
    mx = 0
    my = 2*z - 1
    mz = -2*y + 1
    norm_squared = mx**2 + my**2 + mz**2
    if norm_squared <= 0.05:
        return (1, 0, 0)
    else:
        return (mx, my, mz)

def m_init_vortex(pos): #vortex state
    x, y, z = pos[0]/1e-9, pos[1]/1e-9, pos[2]/1e-9
    mx = 0
    my = np.sin(np.pi/2 * (x-0.5))
    mz = np.cos(np.pi/2 * (x-0.5))
    return (mx, my, mz)

def minimise_system_energy(L, m_init):
    N = 16  # discretisation in one dimension
    cubesize = 100e-9  # cube edge length (m)
    cellsize = cubesize/N  # discretisation in all three dimensions.

    Km = 1e6  # magnetostatic energy density (J/m**3)
    Ms = np.sqrt(2*Km/micromagneticmodel.consts.mu0)  # magnetisation saturation (A/m)
    A = 0.5 * micromagneticmodel.consts.mu0 * Ms**2 * (cubesize/L)**2  # exchange energy constant
    K = 0.1*Km  # Uniaxial anisotropy constant
    u = (0, 0, 1)  # Uniaxial anisotropy easy-axis

    mesh = discretisedfield.Mesh(p1=(0, 0, 0), p2=(cubesize, cubesize, cubesize),
                   cell=(cellsize, cellsize, cellsize))

    system = micromagneticmodel.System(name='stdprob3')
    system.energy = (micromagneticmodel.Exchange(A=A)
                + micromagneticmodel.UniaxialAnisotropy(K=K, u=u)
                + micromagneticmodel.Demag())
    system.m = discretisedfield.Field(mesh, dim=3, value=m_init, norm=Ms)
    md.drive(system, overwrite=True)
    return system


system = minimise_system_energy(8, m_init_vortex)
# system.m.plane('y').mpl()

system = minimise_system_energy(8, m_init_flower)
# system.m.plane('y').mpl()

# phase transition
L_array = np.linspace(8, 9, 5)  # values of L for which the system is relaxed.
vortex_energies = [minimise_system_energy(x, m_init_vortex).table.data['E'].values[0] for x in L_array]
flower_energies = [minimise_system_energy(x, m_init_flower).table.data['E'].values[0] for x in L_array]
plt.plot(L_array, vortex_energies, 'o-', label='vortex')
plt.plot(L_array, flower_energies, 'o-', label='flower')
plt.xlabel('L (lex)')
plt.ylabel('E')
plt.xlim([8.0, 9.0])
plt.grid()
plt.legend()
