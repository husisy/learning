import oommfc
import discretisedfield
import micromagneticmodel
import matplotlib.pyplot as plt

# %matplotlib inline

td = oommfc.TimeDriver()

gamma0 = 2.211e5  # gyromagnetic ratio (m/As)
alpha = 0.1  # Gilbert damping

region = discretisedfield.Region(p1=(0,0,0), p2=(1e-9,1e-9,1e-9))
mesh = discretisedfield.Mesh(region=region, n=(1,1,1))
system = micromagneticmodel.System(name='macrospin')
system.energy = micromagneticmodel.Zeeman(H=(0,0,2e6))
system.dynamics = micromagneticmodel.Precession(gamma0=gamma0) + micromagneticmodel.Damping(alpha=alpha)
system.m = discretisedfield.Field(mesh, dim=3, value=(1,0,0), norm=8e6)
td.drive(system, t=1e-10, n=200)
system.table.data #pandas.core.frame.DataFrame

system.table.data['t'] #time
system.table.data['mz'] #z-component magnetisation
# plt.plot(system.table.data['t'], system.table.data['mz'])
# system.table.data.plot('t', 'mz')
# system.table.data.plot('t', ['mx', 'my', 'mz'])
