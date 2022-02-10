import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
plt.ion()

import kwant

lat_graphene = kwant.lattice.honeycomb()
lat3d = kwant.lattice.general([(0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)], [(0, 0, 0), (0.25, 0.25, 0.25)])

## 2D visualization: graphene quantum dot, edge state
# https://kwant-project.org/doc/1/tutorial/plotting#d-example-zincblende-structure
def make_graphene(r=8, t=-1, tp=-0.1):
    def hf_shape_circle(pos):
        x, y = pos
        ret = x**2 + y**2 < r**2
        return ret
    syst = kwant.Builder()
    syst[lat_graphene.shape(hf_shape_circle, (0, 0))] = 0
    syst[lat_graphene.neighbors()] = t
    syst.eradicate_dangling()
    if tp:
        syst[lat_graphene.neighbors(2)] = tp
    return syst
dev_graphene0 = make_graphene(r=8, t=-1, tp=-0.1)
hf_family_color = lambda site: ('black' if (site.family==lat_graphene.sublattices[0]) else 'white')
hf_hopping_lw = lambda site1,site2: (0.04 if (site1.family==site2.family) else 0.1)
kwant.plot(dev_graphene0, site_lw=0.1, site_color=hf_family_color, hop_lw=hf_hopping_lw)

# compute a wavefunction (number 225) and plot it in different styles
ind_mode = 225
dev_graphene1 = make_graphene(r=8, t=-1, tp=0)
dev_graphene1_f = dev_graphene1.finalized()
wf = np.abs(scipy.linalg.eigh(dev_graphene1_f.hamiltonian_submatrix())[1][:,ind_mode])**2
kwant.plotter.map(dev_graphene1_f, wf, oversampling=10, cmap='gist_heat_r')

# use two different sort of triangles to cleanly fill the space
hf_family_shape = lambda i: (('p',3,180) if (dev_graphene1_f.sites[i].family==lat_graphene.sublattices[0]) else ('p',3,0))
hf_family_color = lambda i: ('black' if dev_graphene1_f.sites[i].family==lat_graphene.sublattices[0] else 'white')
kwant.plot(dev_graphene1_f, site_color=wf, site_symbol=hf_family_shape, site_size=0.5, hop_lw=0, cmap='gist_heat_r')

hf_site_size = lambda i: (3*wf[i]/wf.max())
kwant.plot(dev_graphene1_f, site_size=hf_site_size, site_color=(0,0,1,0.3), hop_lw=0.1)
# kwant.plot(dev_graphene1_f, site_size=3*wf/wf.max(), site_color=(0,0,1,0.3), hop_lw=0.1)


## 3D visualization: zincblende structure (face-centered cubic crystal)
# https://kwant-project.org/doc/1/tutorial/plotting#d-example-zincblende-structure
def make_cuboid(a=15, b=10, c=5):
    def hf_shape_cuboid(pos):
        x, y, z = pos
        ret = (0 <= x < a) and (0 <= y < b) and (0 <= z < c)
        return ret
    syst = kwant.Builder()
    syst[lat3d.shape(hf_shape_cuboid, (0, 0, 0))] = None
    syst[lat3d.neighbors()] = None
    return syst
dev_cuboid0 = make_cuboid(a=15, b=10, c=5)
kwant.plot(dev_cuboid0) #TODO fail on windows

# visualize the crystal structure better for a very small system
dev_cuboid1 = make_cuboid(a=1.5, b=1.5, c=1.5)
hf_family_color = lambda site: ('r' if site.family==lat3d.sublattices[0] else 'g')
kwant.plot(dev_cuboid1, site_size=0.18, site_lw=0.01, hop_lw=0.05, site_color=hf_family_color)
