import os
import numpy as np

import gpaw
import ase
import ase.cluster
import ase.calculators.emt
import ase.optimize

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

# optimise cuboctahedron
z0 = ase.cluster.Octahedron('Ag', 5, cutoff=2)
z0.calc = ase.calculators.emt.EMT()
opt = ase.optimize.BFGS(z0, trajectory=hf_file('opt.traj'))
opt.run(fmax=0.01)

# ground state
# Fermi smearing
z0.calc = gpaw.GPAW(mode='lcao', basis='sz(dzp)', txt=hf_file('gpaw.txt'), occupations=gpaw.FermiDirac(0.1), setups={'Ag': '11'})
z0.center(vacuum=4.0)
z0.get_potential_energy() #-85.23312270728142

# DOS
dos = ase.dft.dos.DOS(z0.calc, npts=800, width=0.1) #width=0.1eV
energies = dos.get_energies()
weights = dos.get_dos()
# plt.close('all')
# plt.plot(energies, weights)
# plt.gcf().savefig(hf_file('cuboctahedron_dos.png'))


## https://wiki.fysik.dtu.dk/gpaw/tutorials/H2/atomization.html
# Hydrogen atom:
tmp0 = [[5,5,5]] #Angstrom
tmp1 = [10,10.0001,10.0002] #break cell symmetry
ase_h = ase.Atoms('H', positions=tmp0, magmoms=[0], cell=tmp1)
#eigensolver='rmm-diis' can parallelize over bands
ase_h.calc = gpaw.GPAW(mode=gpaw.PW(), xc='PBE', hund=True, eigensolver='rmm-diis',
            occupations=gpaw.FermiDirac(0.0, fixmagmom=True), txt=hf_file('H.out'))
energy_h = ase_h.get_potential_energy() #-1.076811971233114
# the energy of a non spin-polarized hydrogen atom is the reference energy

# Hydrogen molecule:
# Experimental bond length: d = 0.74
tmp0 = [[4.63, 5, 5], [5.37,5,5]]
tmp1 = [10,10,10]
ase_h2 = ase.Atoms('H2', positions=tmp0, cell=tmp1)
# No hund rule for molecules
ase_h2.calc = gpaw.GPAW(mode=gpaw.PW(), xc='PBE', hund=False, eigensolver='rmm-diis',
            occupations=gpaw.FermiDirac(0.0, fixmagmom=True), txt=hf_file('H2.out'))
energy_h2 = ase_h2.get_potential_energy() #-6.649255094781373
e_atomization = 2*energy_h - energy_h2 #4.495631152315145


## https://wiki.fysik.dtu.dk/gpaw/tutorials/H2/optimization.html
tmp0 = [[4.63, 5, 5], [5.37,5,5]] # Experimental bond length: d = 0.74
tmp1 = [10,10,10]
ase_h2 = ase.Atoms('H2', positions=tmp0, cell=tmp1)
ase_h2.calc = gpaw.GPAW(mode=gpaw.PW(), xc='PBE', hund=False, eigensolver='rmm-diis',
            occupations=gpaw.FermiDirac(0.0, fixmagmom=True), txt=hf_file('H2.out'))
energy_exp = ase_h2.get_potential_energy() #-6.649255094781373
bond_length_exp = ase_h2.get_distance(0, 1) #0.74

# Find the theoretical bond length:
# ase_h2.set_constraint(ase.constraints.FixAtoms(mask=[False, True])) #error, broken symmetry
relax = ase.optimize.QuasiNewton(ase_h2, logfile=hf_file('qn.log'))
relax.run(fmax=0.05)
energy_pbe_theory = ase_h2.get_potential_energy() #-6.652913343722132
bond_length_pbe_theory = ase_h2.get_distance(0, 1) #0.7549294864200906

##
tmp0 = [[1.95, 2.5, 2.5], [3.05, 2.5, 2.5]]
tmp1 = [5,5,5]
ase_co = ase.Atoms('CO', positions=tmp0, cell=tmp1)

ase_co.calc = gpaw.GPAW(nbands=5, h=0.2, txt=None)
energy = ase_co.get_potential_energy()
ase_co.calc.write(hf_file('CO.gpw'), mode='all') #Save wave functions

num_band = ase_co.calc.get_number_of_bands()
z0 = [ase_co.calc.get_pseudo_wave_function(band=x) for x in range(num_band)] # (list,(np,float64,(24,24,24)))
# Plotting wave functions with jmol/VMD
