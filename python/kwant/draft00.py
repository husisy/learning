import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import kwant


lat_square = kwant.lattice.square()
# lat_graphene = kwant.lattice.general(prim_vecs=[(1,0),(1/2,np.sqrt(3)/2)], basis=[(0,0),(0,1/np.sqrt(3))])
lat_graphene = kwant.lattice.honeycomb() #equivalent, but has attribute .a and .b
lat_kagome = kwant.lattice.kagome()
lat_cubic = kwant.lattice.cubic()
lat_chain = kwant.lattice.chain()


lat_graphene.prim_vecs
lat_graphene.a
lat_graphene.b
lat_graphene.a(2,3).tag #(2,3)
lat_graphene.a(2,3).pos
lat_graphene.b.offset
lat_graphene.b.vec #convert .tag to .pos (without adding offset)


# https://kwant-project.org/doc/1/tutorial/faq#what-is-a-site
dev0 = kwant.Builder()
dev0[lat_square(1,0)] = 4
dev0[lat_square(1,1)] = 4
dev0[(lat_square(1, 0), lat_square(1, 1))] = 1j #(row,column)
# dev0[kwant.builder.HoppingKind((0, -1), lat_square, lat_square)] = 1j #equivalent
# first - second = (0,-1)
dev0_f = dev0.finalized()
hamiltonian = dev0_f.hamiltonian_submatrix() #[[4,1j],[-1j,4]]
# kwant.plot(dev0)
dev0_f.sites #(tuple,kwant.builder.Site,2) (site[1,0], site[1,1])
site_i = dev0_f.sites[0]
site_i.family #lat_square
site_i.tag #(np,int,2) (0,0)
site_i.pos #(np,float,2)
dev0_f.id_by_site[dev0_f.sites[1]] #1
dev0_f.id_by_site[lat_square(1,1)] #1


# https://kwant-project.org/doc/1/tutorial/faq#when-plotting-how-to-color-the-different-sublattices-differently
dev1 = kwant.Builder()
dev1[(lat_kagome.a(i,j) for i in range(4) for j in range(4))] = 4
dev1[(lat_kagome.b(i,j) for i in range(4) for j in range(4))] = 4
dev1[(lat_kagome.c(i,j) for i in range(4) for j in range(4))] = 4
dev1[lat_kagome.neighbors()] = -1
hf_family_color = lambda x: ('#006BA4' if x.family==lat_kagome.a else ('#FF800E' if x.family==lat_kagome.b else '#ABABAB'))
kwant.plot(dev1, site_lw=0.1, site_color=hf_family_color)

z0 = {lat_kagome.a:'a', lat_kagome.b:'b', lat_kagome.c:'c'}
[(x,z0[y],z0[z]) for x,y,z in lat_kagome.neighbors()]
# (tag_vector,target,source): source+tag_vector=target


# Two monatomic lattices
lat_a = kwant.lattice.Monatomic([(1,0), (0,1)], offset=(0,0)) #equivalent to kwant.lattice.square()
lat_b = kwant.lattice.Monatomic([(1,0), (0,1)], offset=(0.5,0.5))
# lat = kwant.lattice.Polyatomic([(1,0),(0,1)], [(0,0),(0.5,0.5)])
# lat_a, lat_b = lat.sublattices
dev2 = kwant.Builder()
dev2[lat_a(0, 0)] = 4
dev2[lat_b(0, 0)] = 4
kwant.plot(dev2)


# https://kwant-project.org/doc/1/tutorial/faq#how-to-create-many-similar-hoppings-in-one-go
dev3 = kwant.Builder()
dev3[(lat_square(i, j) for i in range(5) for j in range(5))] = 4
dev3[kwant.builder.HoppingKind((1, 0), lat_square)] = -1
kwant.plot(dev3)


dev4 = kwant.Builder()
dev4[(lat_kagome.a(i,j) for i in range(4) for j in range(4))] = 4
dev4[(lat_kagome.b(i,j) for i in range(4) for j in range(4))] = 4
dev4[(lat_kagome.c(i,j) for i in range(4) for j in range(4))] = 4
dev4[kwant.builder.HoppingKind((0, 0), lat_kagome.a, lat_kagome.b)] = -1
dev4[kwant.builder.HoppingKind((0, 0), lat_kagome.a, lat_kagome.c)] = -1
dev4[kwant.builder.HoppingKind((0, 0), lat_kagome.c, lat_kagome.b)] = -1
kwant.plot(dev4)


dev5 = kwant.Builder()
dev5[(lat_square(i, j) for i in range(3) for j in range(3))] = 4
dev5[lat_square.neighbors(2)] = -1
kwant.plot(dev5)


# Define the scattering Region
L = 5
W = 5
dev6 = kwant.Builder()
dev6[(lat_graphene.a(i, j) for i in range(L) for j in range(W))] = 4
dev6[(lat_graphene.b(i, j) for i in range(L) for j in range(W))] = 4
dev6[lat_graphene.neighbors()] = -1
lead1 = kwant.Builder(kwant.TranslationalSymmetry((0, 1)))
lead1[(lat_square(i, 0) for i in range(2, 7))] = 4
lead1[lat_square.neighbors()] = -1
dev6[(lat_square(i, 5) for i in range(2, 7))] = 4
dev6[lat_square.neighbors()] = -1
dev6[((lat_square(i+2, 5), lat_graphene.b(i, 4)) for i in range(5))] = -1
dev6.attach_lead(lead1)
kwant.plot(dev6)


# Create 3d model.
def hf_shape_cuboid(site):
    x, y, z = site.pos
    ret = (-4<x<4) and (-10<y<10) and (-3<z<3)
    return ret
def hf_shape_electrode(site):
    x, y, z = site.pos
    ret = (y-5)**2 + (z-2)**2 < 2.3**2
    return ret
cubic = kwant.lattice.cubic()
model = kwant.Builder(kwant.TranslationalSymmetry([1, 0, 0], [0, 1, 0], [0, 0, 1]))
model[lat_cubic(0, 0, 0)] = 4
model[lat_cubic.neighbors()] = -1
dev7 = kwant.Builder()
dev7.fill(model, hf_shape_cuboid, (0, 0, 0))
electrode = kwant.Builder(kwant.TranslationalSymmetry([1, 0, 0]))
electrode.fill(model, hf_shape_electrode, (0,5,2))
hf0 = lambda s: abs(s.pos[0]) < 7
dev7.fill(electrode, hf0, (0, 5, 4))
dev7.attach_lead(electrode)
kwant.plot(dev7) #fail on windows



dev8 = kwant.Builder()
dev8[(lat_graphene.a(i, j) for i in range(6) for j in range(6))] = 4
dev8[(lat_graphene.b(i, j) for i in range(6) for j in range(6))] = 4
dev8[lat_graphene.neighbors()] = -1
kwant.plot(dev8)
lat_graphene_nnn = [
    ((1, 0), lat_graphene.a, lat_graphene.a), #clock
    ((0, 1), lat_graphene.a, lat_graphene.a), #anti-clock
    ((1, -1), lat_graphene.a, lat_graphene.a), #anti-clock
    ((1, 0), lat_graphene.b, lat_graphene.b), #anti-clock
    ((0, 1), lat_graphene.b, lat_graphene.b), #clock
    ((1, -1), lat_graphene.b, lat_graphene.b), #clock
]
lat_graphene.neighbors(2)
z0 = {lat_graphene.a:'a', lat_graphene.b:'b'}
[(x.delta,z0[x.family_a],z0[x.family_b]) for x in lat_graphene.neighbors(2)]



lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
lead[(lat_square(0, i) for i in range(3))] = 4
lead[lat_square.neighbors()] = -1
lead_f = lead.finalized()
prop_modes, _ = lead_f.modes(energy=2.5)
