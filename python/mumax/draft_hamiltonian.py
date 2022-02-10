import oommfc
import discretisedfield
import micromagneticmodel

# %matplotlib inline

md = oommfc.MinDriver() #create energy minimisation driver

p1 = (0, 0, 0)
p2 = (10e-9, 1e-9, 1e-9)
cell = (1e-9, 1e-9, 1e-9)
region = discretisedfield.Region(p1=p1, p2=p2)
mesh = discretisedfield.Mesh(p1=p1, p2=p2, cell=cell)
# mesh.k3d()


# Zeeman energy
system = micromagneticmodel.System(name='zeeman')  # create the (micromagnetic) system object
system.energy = micromagneticmodel.Zeeman(H=(0, 0, 1e6)) # external magnetic field (A/m)
system.m = discretisedfield.Field(mesh, dim=3, value=(1, 0, 1), norm=8e6) # saturation magnetisation (A/m)
# system.m.plane('y').mpl()
# system.m.plane('z').k3d_vector(head_size=3, color_field=system.m.z)
md.drive(system)
# system.energy.zeeman.H = (1e6, 0, 0) # change Zeeman along x direction
# md.drive(system)


# uniaxial anisotropy
system = micromagneticmodel.System(name='uniaxial_anisotropy')
system.energy = micromagneticmodel.UniaxialAnisotropy(K=6e6, u=(1, 0, 1))
def m_initial(pos):
    x, y, z = pos
    ret = (-1,0,-0.1) if (x<=5e-9) else (1,0,0.1)
    return ret
system.m = df.Field(mesh, dim=3, value=m_initial, norm=8e6)
md.drive(system)


# exchange energy
system = micromagneticmodel.System(name='exchange')
system.energy = micromagneticmodel.Exchange(A=8e-12) #exchange energy constant (J/m)
def m_initial(pos):
    x, y, z = pos
    ret = (0, 0, 1) if (x<=5e-9) else (1, 0, 0)
    return ret
system.m = discretisedfield.Field(mesh, dim=3, value=m_initial, norm=8e6)
md.drive(system)


# Dzyaloshinkii-Moriya energy
system = micromagneticmodel.System(name='dmi')
system.energy = micromagneticmodel.DMI(crystalclass='Cnv', D=3e-3) #DMI energy constant (J/m**2)
system.m = discretisedfield.Field(mesh, dim=3, value=(0, 0, 1), norm=8e6)
md.drive(system)


# Zeeman and exchange energy
system = micromagneticmodel.System(name='exchange_and_zeeman')
system.energy = micromagneticmodel.Exchange(A=8e-12) + micromagneticmodel.Zeeman(H=(0, 0, -1e6))
def m_initial(pos):
    x, y, z = pos
    ret = (0, 0, 1) if (x<=5e-9) else (1, 0, 0)
    return ret
system.m = discretisedfield.Field(mesh, dim=3, value=m_initial, norm=8e6)
md.drive(system)


## DMI and exchange energy
system = micromagneticmodel.System(name='exchange_and_DMI')
region = discretisedfield.Region(p1=(0, 0, 0), p2=(20e-9, 1e-9, 1e-9))
mesh = discretisedfield.Mesh(region=region, cell=(1e-9, 1e-9, 1e-9))
system.m = discretisedfield.Field(mesh, dim=3, value=(0,1,1), norm=8e6)
system.energy = micromagneticmodel.Exchange(A=1e-11) + micromagneticmodel.DMI(D=6.28e-3, crystalclass='Cnv')
md.drive(system)
# system.m.plane('y').k3d_vector(color_field=system.m.z)
