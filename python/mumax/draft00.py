import oommfc
import discretisedfield
import micromagneticmodel
# %matplotlib inline #"%matplotlib notebook" will raise error below

system = micromagneticmodel.System(name='first_ubermag_simulation')


# Hamiltonian: exchange, demagnetisation, Zeeman energy
A = 1e-12  # exchange energy constant (J/m)
H = (5e6, 0, 0)  # external magnetic field in the x-direction (A/m)
system.energy = micromagneticmodel.Exchange(A=A) + micromagneticmodel.Demag() + micromagneticmodel.Zeeman(H=H)
system.energy #output latex formula in jupyter

# Dynamics equation: precession, damping
gamma0 = 2.211e5  # gyrotropic ratio parameter (m/As)
alpha = 0.2  # Gilbert damping
system.dynamics = micromagneticmodel.Precession(gamma0=gamma0) + micromagneticmodel.Damping(alpha=alpha)
system.dynamics #output latex formula in jupyter


# Magnetisation configuration
L = 100e-9  # cubic sample edge length (m)
d = 5e-9  # discretisation cell size (m)
mesh = discretisedfield.Mesh(p1=(0, 0, 0), p2=(L, L, L), cell=(d, d, d))
Ms = 8e6  # saturation magnetisation (A/m)
system.m = discretisedfield.Field(mesh, dim=3, value=(0, 1, 0), norm=Ms)


# mesh.k3d() #run in jupyter
# system.m.plane('z').mpl() #run in jupyter
# system.m.plane('z').k3d_vector(head_size=20) #run in jupyter

md = oommfc.MinDriver()
md.drive(system)

system.m.average #magnetisation
system.m.array #(np,float,(Nx,Ny,Nz,3))
system.table.data #pandas.core.frame.DataFrame


micromagneticmodel.Zeeman(H=(0,0,1e6), name='zeeman1') + micromagneticmodel.Zeeman(H=(1e6,0,0), name='zeeman2')
