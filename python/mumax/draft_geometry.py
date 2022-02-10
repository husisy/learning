import oommfc
import discretisedfield
import micromagneticmodel
# %matplotlib inline

L = 100e-9  # edge length (m)
d = 10e-9  # cell size (m)
p1 = (0, 0, 0)  # first point of the cuboid containing simulation geometry
p2 = (L, L, L)  # second point
cell = (d, d, d)  # discretisation cell
region = discretisedfield.Region(p1=p1, p2=p2)
mesh = discretisedfield.Mesh(region=region, cell=cell)  # mesh definition

mesh.region.edges #edge length
mesh.n #number of cells
mesh.region.pmin #minimum mesh domain coordinate
mesh.region.pmax
# mesh.k3d()
# mesh.mpl()


m = discretisedfield.Field(mesh, dim=3, value=(1, 0, 0))
# m.plane('z').mpl()
# m.plane('z').k3d_vector(head_size=30)
# m.norm.k3d_nonzero()


def m_value(pos):
    x, y, z = pos  # unpack position into individual components
    ret = (-1, 1, 0) if (x<=L/5) else (1,0,0)
    return ret
Ms = 8e6  # saturation magnetisation (A/m), default to 1
m = discretisedfield.Field(mesh, dim=3, value=m_value, norm=Ms)
# m.plane('z').mpl()
m((0,0,0)) #position to vector
m((50e-9,0,0)) #position to vector


# spatially varying Ms
mesh = discretisedfield.Mesh(p1=(-L/2, -L/2, -L/2), p2=(L/2, L/2, L/2), cell=(d, d, d))
def Ms_value(pos):
    x, y, z = pos
    ret = Ms if (x**2+y**2+z**2)**0.5<L/2 else 0
    return ret
m = discretisedfield.Field(mesh, dim=3, value=(0, -1, 0), norm=Ms_value)
