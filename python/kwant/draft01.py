import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import kwant

from utils import pauli

lat_square = kwant.lattice.square()
lat_graphene = kwant.lattice.honeycomb()

tmp0 = [
    ((-1, 0), lat_graphene.a, lat_graphene.a),
    ((0, 1), lat_graphene.a, lat_graphene.a),
    ((1, -1), lat_graphene.a, lat_graphene.a),
    ((1, 0), lat_graphene.b, lat_graphene.b),
    ((0, -1), lat_graphene.b, lat_graphene.b),
    ((-1, 1), lat_graphene.b, lat_graphene.b),
]
lat_graphene_neighbor2_anticlock = [kwant.builder.HoppingKind(*x) for x in tmp0]

## Discretizing-Hamiltonians-computing-spectra
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/2.1.Discretizing-Hamiltonians-computing-spectra.ipynb
hf_shape_circle = lambda pos: pos[0]**2 + pos[1]**2 < 15**2
dev1 = kwant.Builder()
dev1[lat_square.shape(hf_shape_circle, (0,0))] = 4
dev1[lat_square.neighbors()] = -1
dev1_f = dev1.finalized()
EVL, EVC = np.linalg.eigh(dev1_f.hamiltonian_submatrix())
kwant.plotter.map(dev1_f, np.abs(EVC[:,0])**2)


## band
param_dev2 = {'width':10, 't':1}
dev2 = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
dev2[(lat_square(0, y) for y in range(param_dev2['width']))] = 4*param_dev2['t']
dev2[lat_square.neighbors()] = -param_dev2['t']
# dev2[lat_square(0,0), lat_square(0,param_dev2['width']-1)] = -param_dev2['t'] #make it periodic
dev2_f = dev2.finalized()
kwant.plotter.bands(dev2_f)


## perfect transmission
param_wire = {'width':10, 'length':30, 't':1}
param_wire['width']
dev_wire = kwant.Builder()
dev_wire[(lat_square(x,y) for x in range(param_wire['length']) for y in range(param_wire['width']))] = 4*param_wire['t']
dev_wire[lat_square.neighbors()] = -param_wire['t']
lead = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
lead[(lat_square(0, y) for y in range(param_wire['width']))] = 4*param_wire['t']
lead[lat_square.neighbors()] = -param_wire['t']
dev_wire.attach_lead(lead)
dev_wire.attach_lead(lead.reversed())
dev_wire_f = dev_wire.finalized()
energy = np.linspace(0,4,100)
tmp0 = [kwant.smatrix(dev_wire_f, x) for x in energy]
transmission = np.array([x.transmission(1,0) for x in tmp0])
reflection = np.array([x.transmission(0,0) for x in tmp0])
fig,ax = plt.subplots()
ax.plot(energy, transmission, label='transmission')
ax.plot(energy, reflection, label='reflection')
ax.legend()


## quantum dot
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/2.2.scattering.ipynb
def hf_shape_dot(pos):
    ret = (pos[0]-param_dot['center'][0])**2 + (pos[1]-param_dot['center'][1])**2 < param_dot['radius']**2
    return ret
param_dot = {
    'center': (3,7),
    'radius': 13,
    't': 1,
}
dev_dot = kwant.Builder()
dev_dot[lat_square.shape(hf_shape_dot, param_dot['center'])] = 4 * param_dot['t']
dev_dot[lat_square.neighbors()] = -param_dot['t']
lead = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
lead[(lat_square(0, y) for y in range(param_wire['width']))] = 4*param_wire['t']
lead[lat_square.neighbors()] = -param_wire['t']
dev_dot.attach_lead(lead)
dev_dot.attach_lead(lead.reversed())
dev_dot_f = dev_dot.finalized()
energy = np.linspace(0,3,100) #singular at E=4
tmp0 = [kwant.smatrix(dev_dot_f, x) for x in energy]
transmission = np.array([x.transmission(1,0) for x in tmp0])
reflection = np.array([x.transmission(0,0) for x in tmp0])
fig,ax = plt.subplots()
ax.plot(energy, transmission, label='transmission')
ax.plot(energy, reflection, label='reflection')
ax.legend()


## transport through barrier
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/2.3.transport_through_barrier.ipynb
param_wire1 = {
    'width': 10,
    'length': 2,
    't': 1,
}
energy = 1
voltage = np.linspace(-2, 2, 21)
dev_wire1 = kwant.Builder()
hf_onsite = lambda site,voltage: (4-voltage)*param_wire1['t']
dev_wire1[(lat_square(x, y) for x in range(param_wire1['length']) for y in range(param_wire1['width']))] = hf_onsite
dev_wire1[lat_square.neighbors()] = -param_wire1['t']
lead = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
lead[(lat_square(0, y) for y in range(param_wire1['width']))] = 4 * param_wire1['t']
lead[lat_square.neighbors()] = -param_wire1['t']
dev_wire1.attach_lead(lead)
dev_wire1.attach_lead(lead.reversed())
dev_wire1_f = dev_wire1.finalized()
transmission = [kwant.smatrix(dev_wire1_f, energy, params={'voltage':x}).transmission(1, 0) for x in voltage]
fig,ax = plt.subplots()
ax.plot(voltage, transmission)
ax.set_xlabel('voltage')
ax.set_ylabel('transmission')


## transport through barrier
# J. Appl. Phys. 77, 4504 (1995): http://dx.doi.org/10.1063/1.359446
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/2.3.transport_through_barrier.ipynb
def onsite_with_gate(site, voltage):
    x,y = site.pos
    d1,l1,r1,b1,t1 = param_wire2['gate1_dlrbt']
    d2,l2,r2,b2,t2 = param_wire2['gate2_dlrbt']
    hf1 = lambda dx,dy,dz: np.arctan2(dx*dy, dz*np.sqrt(dx**2 + dy**2 + dz**2)) / (2*np.pi)
    tmp1 = hf1(x-l1,y-b1,d1) + hf1(x-l1,t1-y,d1) + hf1(r1-x,y-b1,d1) + hf1(r1-x,t1-y,d1)
    tmp2 = hf1(x-l2,y-b2,d2) + hf1(x-l2,t2-y,d2) + hf1(r2-x,y-b2,d2) + hf1(r2-x,t2-y,d2)
    ret = 4*param_wire2['t'] - voltage*tmp1 - voltage*tmp2
    return ret
param_wire2 = {
    'width': 40,
    'length': 70,
    't': 1,
    'gate1_dlrbt': (10, 20, 50, -50, 15), #distance, left, right, bottom, top
    'gate2_dlrbt': (10, 20, 50, 25, 90),
}
energy = 0.3
voltage = np.linspace(-0.8, 0, 51)
dev_wire2 = kwant.Builder()
dev_wire2[(lat_square(x, y) for x in range(param_wire2['length']) for y in range(param_wire2['width']))] = onsite_with_gate
dev_wire2[lat_square.neighbors()] = -param_wire2['t']
lead = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
lead[(lat_square(0, y) for y in range(param_wire2['width']))] = 4 * param_wire2['t']
lead[lat_square.neighbors()] = -param_wire2['t']
dev_wire2.attach_lead(lead)
dev_wire2.attach_lead(lead.reversed())
# kwant.plotter.map(dev_wire2, lambda s: onsite_with_gate(s, -0.5))
dev_wire2_f = dev_wire2.finalized()
transmission = [kwant.smatrix(dev_wire2_f, energy, params={'voltage':x}).transmission(1, 0) for x in voltage]
fig,ax = plt.subplots()
ax.plot(voltage, transmission)
ax.set_xlabel('voltage')
ax.set_ylabel('transmission')


## Aharonov-Bohm-cheat
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/3.1.Aharonov-Bohm-cheat.ipynb
def hf_shape_ring(pos):
    x,y = pos
    tmp0 = x**2 + y**2
    ret = (tmp0>param_ring['r0']**2) and (tmp0<param_ring['r1']**2)
    return ret
def hf_shape_hops_across_cut(dev):
    ret = []
    tmp0 = kwant.builder.HoppingKind((1, 0), lat_square, lat_square)(dev) #hop1+(1,0)=hop0
    for hop0,hop1 in tmp0:
        if hop0.tag[0]==1 and hop0.tag[1]>0 and hop1.tag[0]==0:
            ret.append((hop0,hop1))
    return ret
def hf_shape_lead(pos):
    ret = abs(pos[1])<(param_ring['r1']-param_ring['r0'])/2
    return ret
param_ring = {
    'r0': 22,
    'r1': 38,
}
energy = 3.3
phis = np.linspace(0, 1, 50)
dev_ring = kwant.Builder()
dev_ring[lat_square.shape(hf_shape_ring, (param_ring['r0']+1,0))] = 4
dev_ring[lat_square.neighbors()] = 1
dev_ring[hf_shape_hops_across_cut] = lambda site0,site1,phi: np.exp(-2j*np.pi*phi)
lead = kwant.Builder(kwant.TranslationalSymmetry(lat_square.vec((1,0))))
hf0 = lambda xy: ((abs(xy[1])<param_ring['width']/2) and (abs(xy[0])<3))
lead[lat_square.shape(hf_shape_lead, (0,0))] = 4
lead[lat_square.neighbors()] = 1
dev_ring.attach_lead(lead)
dev_ring.attach_lead(lead.reversed())
# kwant.plot(dev_ring)
dev_ring_f = dev_ring.finalized()
transmission = [kwant.smatrix(dev_ring_f, energy, params={'phi':x}).transmission(1,0) for x in phis]
fig,ax = plt.subplots()
ax.plot(phis, transmission)
ax.set_xlabel('phi = BS/(h/e)')
ax.set_ylabel('conductance ($2e^2/h$)')
ax.set_title('Aharonov-Effect')



def hf_shape_ring1(pos):
    x,y = pos
    tmp0 = x**2 + y**2
    ret = (tmp0>param_ring1['r0']**2) and (tmp0<param_ring1['r1']**2)
    return ret
def hopping_field(site1, site2, phi):
    x1,y1 = site1.pos
    x2,y2 = site2.pos
    ret = -np.exp(-0.5j * phi * (x1 - x2) * (y1 + y2))
    return ret
def hf_shape_lead(pos):
    ret = abs(pos[1])<(param_ring1['r1']-param_ring1['r0'])/2
    return ret
param_ring1 = {
    'length': 100,
    'width': 12,
    'r0': 94,
    'r1': 106,
}
phis = np.linspace(0, 0.0005, 50)
energy = 3.3
dev_ring1 = kwant.Builder()
dev_ring1[lat_square.shape(hf_shape_ring1, (param_ring1['length'],0))] = 4
dev_ring1[lat_square.neighbors()] = hopping_field
lead = kwant.Builder(kwant.TranslationalSymmetry([-1,0]))
lead[lat_square.shape(hf_shape_lead, (0,0))] = 4
lead[lat_square.neighbors()] = hopping_field
dev_ring1.attach_lead(lead)
dev_ring1.attach_lead(lead.reversed())
# kwant.plot(dev_ring1)
dev_ring1_f = dev_ring1.finalized()
transmission = [kwant.smatrix(dev_ring1_f, energy, params={'phi':x}).transmission(1,0) for x in phis]
fig,ax = plt.subplots()
ax.plot(phis, transmission)
ax.set_xlabel('phi = Ba^2/(h/e)')
ax.set_ylabel('g in unit of (2e^2/h)')
ax.set_title('Aharonov-Effect')


## Quantum-Hall-effect-and-disorder
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/3.2.Quantum-Hall-effect-and-disorder.ipynb
def hf_hopping(site1, site2, phi):
    x1,y1 = site1.pos
    x2,y2 = site2.pos
    ret = -np.exp(-0.5j * phi * (x1-x2) * (y1+y2))
    return ret
param_wire3 = {
    't': 1,
    'U0': 0.3,
    'length': 50,
    'width': 30,
    'energy': 3.3,
}
hf_onsite_central = lambda site: np.random.uniform(-0.5,0.5)*param_wire3['U0'] + 4*param_wire3['t']
energy = 3.3
phis = np.linspace(0, 0.1, 51)
dev_wire3 = kwant.Builder()
dev_wire3[(lat_square(x, y) for x in range(param_wire3['length']) for y in range(param_wire3['width']))] = hf_onsite_central
dev_wire3[lat_square.neighbors()] = hf_hopping
lead = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
lead[(lat_square(0, y) for y in range(param_wire3['width']))] = 4 * param_wire3['t']
lead[lat_square.neighbors()] = hf_hopping
dev_wire3.attach_lead(lead)
dev_wire3.attach_lead(lead.reversed())
dev_wire3_f = dev_wire3.finalized()

## plot wave function
# tmp1 = kwant.wave_function(dev_wire3_f, energy=energy, params={'phi':0.07})
# wave_function = tmp1(0) #wave function from lead 0
# kwant.plotter.map(dev_wire3_f, np.sum(np.abs(wave_function)**2, axis=0))

transmission = [kwant.smatrix(dev_wire3_f, energy, params={'phi':x}).transmission(1,0) for x in phis]
fig,ax = plt.subplots()
ax.plot(phis, transmission)
ax.set_xlabel('phi = Ba^2/(h/e)')
ax.set_ylabel('g in unit of (2e^2/h)')


## Magneto Resistance
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/3.3.MagnetoResistance-cheat.ipynb
def onsite_central(site, angle):
    x,_ = site.pos
    EExchange = param_wire4['EExchange']
    width = param_wire4['width']
    if (width<x) and (x<2*width):
        return np.array([[4+EExchange,0], [0,4-EExchange]])
    elif (3*width<x) and (x<4*width):
        # 4*PauliX + EExchange*cos(angle)*PauliZ + EExchange*sin(angle)*PauliX
        tmp1 = 4 + EExchange*np.cos(angle)
        tmp2 = EExchange*np.sin(angle)
        tmp3 = 4 - EExchange*np.cos(angle)
        return np.array([[tmp1, tmp2], [tmp2,tmp3]])
    else:
        return np.array([[4,0], [0,4]])
param_wire4 = {'width':10, 'EExchange':0.2}
energy = 2.3
angles = np.linspace(0,2*np.pi,100)
dev_wire4 = kwant.Builder()
dev_wire4[(lat_square(x,y) for x in range(5*param_wire4['width']) for y in range(param_wire4['width']))] = onsite_central
dev_wire4[lat_square.neighbors()] = np.array([[1,0], [0,1]])
lead = kwant.Builder(kwant.TranslationalSymmetry(lat_square.vec((1,0))))
lead[(lat_square(0,y) for y in range(param_wire4['width']))] = np.array([[4,0], [0,4]])
lead[lat_square.neighbors()] = np.array([[1,0], [0,1]])
dev_wire4.attach_lead(lead)
dev_wire4.attach_lead(lead.reversed())
dev_wire4_f = dev_wire4.finalized()
transmission = [kwant.smatrix(dev_wire4_f, energy, params={'angle':x}).transmission(1,0) for x in angles]
fig,ax = plt.subplots()
ax.plot(angles, transmission)
ax.set_xlabel('angle')
ax.set_ylabel('transmission')


## hall resistance
# https://nbviewer.jupyter.org/github/kwant-project/kwant-tutorial-2016/blob/master/3.3.MagnetoResistance-cheat.ipynb
def hf_hall_resistance(smatrix):
    G = np.array([[smatrix.transmission(i,j) for j in range(4)] for i in range(4)], dtype=np.float64)
    G[range(4),range(4)] = G.diagonal() - G.sum(axis=1)
    V = np.linalg.solve(G[:3,:3], np.array([1,0,0]))
    ret = V[2]-V[1]
    return ret
def hf_onsite_central(site, delta):
    radius0 = param['radius0']
    EExchange = param['EExchange']
    x,y = site.pos
    r = (x**2 + y**2)**0.5
    theta = (np.tanh((radius0-r)/delta)+1) * np.pi / 2
    if r != 0:
        tmp1 = x*np.sin(theta)/r
        tmp2 = y*np.sin(theta)/r
        tmp3 = np.cos(theta)
        ret = np.array([[4,0], [0,4]]) + EExchange*np.array([[tmp3,tmp1-1j*tmp2], [tmp1+1j*tmp2,-tmp3]])
    else:
        ret = np.array([[4+EExchange,0], [0,4-EExchange]])
    return ret
param = {
    'length': 20,
    'width': 15,
    'EExchange': 1,
    'radius0': 6,
}
energy = 2
deltas = np.linspace(0.1, 3, 50)
dev_bar = kwant.Builder()
tmp1 = {(x,y) for x in range(-param['length']+1,param['length']) for y in range(-param['width']+1,param['width'])}
tmp2 = {(x,y) for x in range(-param['width']+1,param['width']) for y in range(-param['length']+1,param['length'])}
dev_bar[(lat_square(x,y) for x,y in tmp1.union(tmp2))] = hf_onsite_central
dev_bar[lat_square.neighbors()] = -pauli.s0
lead_l = kwant.Builder(kwant.TranslationalSymmetry((-1,0)))
lead_l[(lat_square(0,y) for y in range(-param['width']+1,param['width']))] = pauli.sz*param['EExchange'] + 4*pauli.s0
lead_l[lat_square.neighbors()] = -pauli.s0
lead_d = kwant.Builder(kwant.TranslationalSymmetry((0,-1)))
lead_d[(lat_square(x,0) for x in range(-param['width']+1,param['width']))] = pauli.sz*param['EExchange'] + 4*pauli.s0
lead_d[lat_square.neighbors()] = -pauli.s0
dev_bar.attach_lead(lead_l)
dev_bar.attach_lead(lead_d)
dev_bar.attach_lead(lead_d.reversed())
dev_bar.attach_lead(lead_l.reversed())
dev_bar_f = dev_bar.finalized()
hall_resistance = [hf_hall_resistance(kwant.smatrix(dev_bar_f, energy, params={'delta':x})) for x in deltas]
fig,ax = plt.subplots()
ax.plot(deltas, hall_resistance)
ax.set_xlabel('deltas')
ax.set_ylabel('hall resistance')
fig.tight_layout()


# https://nbviewer.org/github/kwant-project/kwant-tutorial-2016/blob/master/3.4.graphene_qshe.ipynb
zigzag_ribbon = kwant.Builder(kwant.TranslationalSymmetry([1, 0]))
zigzag_ribbon[lat_graphene.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = 0
zigzag_ribbon[lat_graphene.neighbors(1)] = 1
kwant.plotter.bands(zigzag_ribbon.finalized())

armchair_ribbon = kwant.Builder(kwant.TranslationalSymmetry([0, np.sqrt(3)]))
armchair_ribbon[lat_graphene.shape((lambda pos: abs(pos[0]) < 9), (0, 0))] = 0
armchair_ribbon[lat_graphene.neighbors(1)] = 1
kwant.plotter.bands(armchair_ribbon.finalized())

zigzag_ribbon = kwant.Builder(kwant.TranslationalSymmetry([1, 0]))
zigzag_ribbon[lat_graphene.a.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = 0.2
zigzag_ribbon[lat_graphene.b.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = -0.2
zigzag_ribbon[lat_graphene.neighbors(1)] = 1
kwant.plotter.bands(zigzag_ribbon.finalized())


param_haldane = {'t_2': 0.05, 'm':0.2}
dev_haldane = kwant.Builder(kwant.TranslationalSymmetry([1, 0]))
dev_haldane[lat_graphene.a.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = param_haldane['m']
dev_haldane[lat_graphene.b.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = -param_haldane['m']
dev_haldane[lat_graphene.neighbors(1)] = 1
dev_haldane[lat_graphene_neighbor2_anticlock] = 1j*param_haldane['t_2']
kwant.plotter.bands(dev_haldane.finalized())

param_kane_mele = {'t_2':0.05, 'm':0.2}
dev_kane_mele = kwant.Builder(kwant.TranslationalSymmetry([1, 0]))
dev_kane_mele[lat_graphene.a.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = param_kane_mele['m']*pauli.s0
dev_kane_mele[lat_graphene.b.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = -param_kane_mele['m']*pauli.s0
dev_kane_mele[lat_graphene.neighbors()] = pauli.s0
dev_kane_mele[lat_graphene_neighbor2_anticlock] = 1j * param_kane_mele['t_2'] * pauli.sz
kwant.plotter.bands(dev_kane_mele.finalized())

param_kane_mele1 = {'t_2':0.05, 'm':0.2}
dev_kane_mele1 = kwant.Builder(kwant.TranslationalSymmetry([1, 0]))
dev_kane_mele1[lat_graphene.a.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = param_kane_mele1['m']*pauli.s0
dev_kane_mele1[lat_graphene.b.shape((lambda pos: abs(pos[1]) < 9), (0, 0))] = -param_kane_mele1['m']*pauli.s0
dev_kane_mele1[lat_graphene.neighbors()] = pauli.s0
dev_kane_mele1[kwant.builder.HoppingKind((0, 0), lat_graphene.b, lat_graphene.a)] = pauli.s0 + 0.3j * pauli.sx
dev_kane_mele1[lat_graphene_neighbor2_anticlock] = 1j * param_kane_mele1['t_2'] * pauli.sz
kwant.plotter.bands(dev_kane_mele1.finalized())
