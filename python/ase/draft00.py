import os
import numpy as np

import ase
import ase.io
import ase.build
import ase.optimize
import ase.spacegroup
import ase.calculators.emt
import ase.md.verlet

import gpaw

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


tmp0 = [3.55*np.sqrt(2), 3.55*np.sqrt(2), 1, 90, 90, 120]
tmp1 = [(0,0,0), (0.5,0,0), (0,0.5,0), (0.5,0.5,0)]
mol_ni4 = ase.Atoms('Ni4', cell=tmp0, pbc=[True,True,False], scaled_positions=tmp1)
mol_ni4.center(vacuum=5, axis=2)
mol_ni4.cell #(ase.cell.Cell)
mol_ni4.positions #(np,float)
mol_ni4[0] #(ase.atom.Atom)
mol_ni4.get_scaled_positions() #(np,float)
mol_ni4.symbols #(str)'Ni4'
mol_ni4.pbc #(np,bool)[True,True,False]
# import ase.visualize
# ase.visualize.view(mol_ni4)


## nitrogen (N2) on copper (Cu)
slab_cu = ase.build.fcc111('Cu', size=(4, 4, 2), vacuum=10.0)
slab_cu.calc = ase.calculators.emt.EMT()
e_slab = slab_cu.get_potential_energy() #11.509056283570382

mol_n2 = ase.Atoms('2N', positions=[(0,0,0), (0,0,1.10)])
mol_n2.calc = ase.calculators.emt.EMT()
e_N2 = mol_n2.get_potential_energy() #0.44034357303561467

ase.build.add_adsorbate(slab_cu, mol_n2, height=1.85, position='ontop')
slab_cu.set_constraint(ase.constraints.FixAtoms(mask=[x.symbol!='N' for x in slab_cu]))
dyn = ase.optimize.QuasiNewton(slab_cu, trajectory=hf_file('tbd00.traj'))
dyn.run(fmax=0.05)
e_slab_N2 = slab_cu.get_potential_energy() #11.625880434287913 eV
e_absorption = e_slab + e_N2 - e_slab_N2 #0.3235194223180837
force = slab_cu.get_forces() #eV/Angstrom

dyn = ase.md.verlet.VelocityVerlet(mol_n2, timestep=ase.units.fs)
for ind0 in range(10):
    print(f'[step={ind0}] potential={mol_n2.get_potential_energy():.4f}, kinetic={mol_n2.get_kinetic_energy():.4f}')
    dyn.run(steps=20)

ase.io.write(hf_file('tbd00.xyz'), slab_cu)
mol_cu_n2 = ase.io.read(hf_file('tbd00.xyz'))
# ase gui tbd00/tbd00.xyz

# traj_manager = ase.io.trajectory.Trajectory(hf_file('tbd00.traj'), 'w')
# traj_manager.write(mol_cu_n2)

## binding curve of N2 (10 seconds)
distance_list = np.linspace(1, 1.4, 10)
energy_list = []
for distance_i in distance_list:
    tmp0 = distance_i/2
    mol_n2 = ase.Atoms('N2', positions=[[0, 0,-distance_i/2], [0,0,distance_i/2]]) #Angstrom
    mol_n2.center(vacuum=3.0)
    mol_n2.calc = gpaw.GPAW(mode='lcao', basis='dzp', txt=hf_file('gpaw.txt'), xc='LDA')
    energy_list.append(mol_n2.get_potential_energy())
energy_list = np.array(energy_list)
#min_distance=1.1333333333333333, min_energy=-15.76766087243421
# import matplotlib.pyplot as plt
# plt.ion()
# fig,ax = plt.subplots()
# ax.plot(distance_list, energy_list)

mol_n = ase.Atoms('N')
mol_n.center(vacuum=3.0)
mol_n.set_initial_magnetic_moments([3]) #spin polarized
calc = gpaw.GPAW(mode='lcao', basis='dzp', txt=hf_file('gpaw.txt'), xc='LDA')
mol_n.calc = calc
e_n = mol_n.get_potential_energy()
e_binding = energy_list.min() - 2*e_n

## atom manipulation
tmp0 = [3.55*np.sqrt(2), 3.55*np.sqrt(2), 1, 90, 90, 120]
tmp1 = [(0,0,0), (0.5,0,0), (0,0.5,0), (0.5,0.5,0)]
mol_ni_ag = ase.Atoms('Ni4', cell=tmp0, pbc=(True,True,False), scaled_positions=tmp1)
mol_ni_ag.center(vacuum=5, axis=2)
mol_ni_ag.append('Ag')
mol_ni_ag.positions[-1] = np.dot(np.array([1/6,1/6,0.5]), mol_ni_ag.cell) + np.array([0,0,1.9])

tmp0 = np.array([
        [0.27802511, -0.07732213, 13.46649107],
        [0.91833251, -1.02565868, 13.41456626],
        [0.91865997, 0.87076761, 13.41228287],
        [1.85572027, 2.37336781, 13.56440907],
        [3.13987926, 2.3633134, 13.4327577],
        [1.77566079, 2.37150862, 14.66528237],
        [4.52240322, 2.35264513, 13.37435864],
        [5.16892729, 1.40357034, 13.42661052],
        [5.15567324, 3.30068395, 13.4305779],
        [6.10183518, -0.0738656, 13.27945071],
        [7.3856151, -0.07438536, 13.40814585],
        [6.01881192, -0.08627583, 12.1789428],
])
tmp1 = np.array([[8.490373,0,0], [0,4.901919,0], [0,0,26.93236]])
mol_h2o = ase.Atoms('4(OH2)', positions=tmp0, cell=tmp1, pbc=[1, 1, 0])
mol_ni243 = ase.build.fcc111('Ni', size=[2,4,3], a=3.55, orthogonal=True)


## structure optimization
mol_h2o = ase.Atoms('HOH', positions=[[0,0,-1], [0,1,0], [0,0,1]])
mol_h2o.center(vacuum=3.0)
mol_h2o.calc = gpaw.GPAW(mode='lcao', basis='dzp', txt=hf_file('gpaw.txt'))
opt = ase.optimize.BFGS(mol_h2o, trajectory=hf_file('opt.traj'), logfile=hf_file('opt.log'))
opt.run(fmax=0.05)
# ase gui tbd00/opt.traj


## g2 molecule dataset
mol_h2o = ase.build.molecule('H2O', vacuum=3)
len(ase.collections.g2) #162
for x in ase.collections.g2:
    print(x.symbols)
ase.collections.g2['CH3CH2OH']


## https://wiki.fysik.dtu.dk/ase/gettingstarted/tut04_bulk/bulk.html
ase_ag = ase.build.bulk('Ag')
ase_ag.calc = gpaw.GPAW(mode=gpaw.PW(350), kpts=[8,8,8], txt=hf_file('gpaw.bulk.Ag.txt'), setups={'Ag':'11'})
# 11 means 11-electron PAW instead of 17-electrons(default)
energy = ase_ag.get_potential_energy() #-3.6672943924564407
# ase_ag.calc.write(hf_file('tbd00.gpaw'))
# new_calc = gpaw.GPAW(hf_file('tbd00.gpaw'))

# DOS
dos = ase.dft.dos.DOS(ase_ag.calc, npts=800, width=0) #width=0: linear tetrahedron interpolation method
energies = dos.get_energies() #(np,float,800) eV, the zero point of the energy axis is the Fermi energy
weights = dos.get_dos() #(np,float,800) 1/eV
# fig,ax = plt.subplots()
# ax.plot(energies, weights)
# TODO s/p/d-projected DOS

# band
lat = ase_ag.cell.get_bravais_lattice()
print(lat.description()) #special point names GKLUWX etc.
# lat.plot_bz(show=True)
path = ase_ag.cell.bandpath('WLGXWK', density=10) #(ase.dft.kpoints.BandPath)
path.write(hf_file('ag_path.json')) #ase reciprocal path.json
ase_ag.calc.set(kpts=path, fixdensity=True, symmetry='off')
ase_ag.get_potential_energy() #necessary since ase_ag.calc.set
bs = ase_ag.calc.band_structure() #ase.spectrum.band_structure.BandStructure
bs.write(hf_file('ag_bs.json')) #ase band-structure ag_bs.json


## Rutile TiO2 optimization (1 minute)
a = 4.6
c = 2.95
ase_tio2 = ase.spacegroup.crystal(['Ti', 'O'], basis=[(0, 0, 0), (0.3, 0.3, 0.0)],
                spacegroup=136, cellpar=[a, a, c, 90, 90, 90])
ase_tio2.calc = gpaw.GPAW(mode=gpaw.PW(800), kpts=[2,2,3], txt=hf_file('gpaw.rutile.txt'))
opt = ase.optimize.BFGS(ase.constraints.ExpCellFilter(ase_tio2))
opt.run(fmax=0.05)
ase_tio2.cell.get_bravais_lattice() #(ase.lattice.TET) TET(a=4.5492195746015422486, c=2.9293315277229678983)

# DOS
dos = ase.dft.dos.DOS(ase_tio2.calc, npts=800, width=0) #width=0: linear tetrahedron interpolation method
energies = dos.get_energies() #(np,float,800) eV
weights = dos.get_dos() #(np,float,800) 1/eV
# plt.close('all')
# plt.plot(energies, weights)
# plt.gcf().savefig(hf_file('tio2_dos.png'))


# band
path = ase_tio2.cell.bandpath(density=7)
path.write(hf_file('rutile_path.json'))
ase_tio2.calc.set(kpts=path, fixdensity=True, symmetry='off')
ase_tio2.get_potential_energy() #-57.193943816443095
bs = ase_tio2.calc.band_structure()
bs.write(hf_file('rutile_bs.json'))

# TODO EOS https://wiki.fysik.dtu.dk/ase/gettingstarted/tut04_bulk/bulk.html#equation-of-state
# TODO nanoparticle https://wiki.fysik.dtu.dk/ase/gettingstarted/cluster/cluster.html
