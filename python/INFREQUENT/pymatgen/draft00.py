import pymatgen as mg
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer #integrated from spglib


x1 = mg.Element('Si')
x1.atomic_mass
x1.melting_point
print(x1.melting_point)


x1 = mg.Composition('Fe2O3')
x1.weight
x1['Fe']
x1.get_atomic_fraction('Fe')


x1 = mg.Lattice.cubic(4.2)
x2 = mg.Structure(x1, ['Cs','Cl'], [[0,0,0],[0.5,0.5,0.5]])
x2.volume
SpacegroupAnalyzer(x2).get_space_group_symbol()
x2.to(fmt="poscar")
# x3 = mg.Structure.from_str(x2.to(fmt='poscar'), fmt='poscar')

x1 = mg.Structure.from_spacegroup('Fm-3m', mg.Lattice.cubic(3), ['Li','O'], [[0.25,0.25,0.25],[0,0,0]])
