# https://kwant-project.org/doc/1/tutorial/
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import kwant

tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]

lat_graphene = kwant.lattice.honeycomb()


## graphene lattice, pn-junction
# https://kwant-project.org/doc/1/tutorial/graphene
def hf_shape_circle(pos):
    x, y = pos
    ret = x**2+y**2 < param_graphene['radius']**2
    return ret
def potential(site):
    (x, y) = site.pos
    d = y*np.sqrt(3)/2 + x/2
    # distance to the line "x+sqrt(3)*y=0"
    ret = param_graphene['pot'] * np.tanh(d / param_graphene['width'])
    return ret
def hf_shape_lead0(pos):
    x, y = pos
    ret = (-0.4*param_graphene['radius']<y) and (y<0.4*param_graphene['radius'])
    return ret
def hf_shape_lead1(pos):
    v = pos[1]/2 - pos[0]*np.sqrt(3)/2
    ret = (-0.4*param_graphene['radius']<v) and (v<0.4*param_graphene['radius'])
    return ret
param_graphene = {
    'radius': 10,
    'width': 2,
    'pot': 0.1,
}
dev_graphene = kwant.Builder()
dev_graphene[lat_graphene.shape(hf_shape_circle, (0, 0))] = potential
# lat_graphene.a.shape(hf_shape_circle, (0,0))
hoppings = (
    ((0, 0), lat_graphene.a, lat_graphene.b), #(i,j), target, source
    ((0, 1), lat_graphene.a, lat_graphene.b),
    ((-1, 1), lat_graphene.a, lat_graphene.b)
)
dev_graphene[[kwant.builder.HoppingKind(*x) for x in hoppings]] = -1
del dev_graphene[lat_graphene.a(0, 0)]
dev_graphene[lat_graphene.a(-2, 1), lat_graphene.b(2, 2)] = -1
lead0 = kwant.Builder(kwant.TranslationalSymmetry(lat_graphene.vec((-1, 0))))
lead0[lat_graphene.shape(hf_shape_lead0, (0, 0))] = -param_graphene['pot']
lead0[[kwant.builder.HoppingKind(*x) for x in hoppings]] = -1
lead1 = kwant.Builder(kwant.TranslationalSymmetry(lat_graphene.vec((0, 1))))
lead1[lat_graphene.shape(hf_shape_lead1, (0, 0))] = param_graphene['pot']
lead1[[kwant.builder.HoppingKind(*x) for x in hoppings]] = -1
dev_graphene.attach_lead(lead0)
dev_graphene.attach_lead(lead1)
dev_graphene_f = dev_graphene.finalized()

hf_family_colors = lambda x: (0 if x.family==lat_graphene.a else 1)
kwant.plot(dev_graphene, site_color=hf_family_colors, site_lw=0.1, lead_site_lw=0, colorbar=False)

# band structure of lead0
kx = np.linspace(-np.pi, np.pi, 100)
tmp0 = kwant.physics.Bands(dev_graphene_f.leads[0])
energy_band = np.stack([tmp0(k) for k in kx])
fig,ax = plt.subplots()
ax.plot(kx, energy_band, color=tableau_colorblind[1])
ax.set_xlabel('kx')
ax.set_title('energy band')

# conductance
energy = np.linspace(-2*param_graphene['pot'], 2*param_graphene['pot'], 51)
transmission = np.array([kwant.smatrix(dev_graphene_f, x).transmission(0, 1) for x in energy])
fig,ax = plt.subplots()
ax.plot(energy, transmission, color=tableau_colorblind[1])
ax.set_xlabel('energy (t)')
ax.set_ylabel('conductance ($e^2/h$)')
