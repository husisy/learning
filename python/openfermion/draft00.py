import numpy as np

from openfermion.hamiltonians import fermi_hubbard, jellium_model, MolecularData
from openfermion.ops import FermionOperator, QubitOperator
from openfermion.transforms import get_sparse_operator, jordan_wigner, bravyi_kitaev, reverse_jordan_wigner, get_fermion_operator
from openfermion.utils import (commutator, count_qubits, hermitian_conjugated,
        normal_ordered, eigenspectrum, get_ground_state, fourier_transform, Grid)

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

FermionOperator() #zero
FermionOperator(()) #one
FermionOperator([(3,1), (1,0)])
FermionOperator('3^ 1')
FermionOperator([(3,1), (1,0)], 2.33)
FermionOperator('3^ 1', 2.33)

z0 = FermionOperator('3^ 1', 2.33)
z0.terms
get_sparse_operator(z0)

# supported operator: == != = / /= + += minus- -= negative- **
FermionOperator('4^ 3^ 9 1', 1+2j) + FermionOperator('3^ 1', -1.7)

z0 = FermionOperator('4^ 3 3^', 1+2j)
hermitian_conjugated(z0)
z0.is_normal_ordered()
count_qubits(z0)
z1 = normal_ordered(z0)
commutator(z0, z1) #commutator(a,b) = a*b + b*a


z0 = QubitOperator('X1 Y2 Z3') #always sorted
z0.terms
get_sparse_operator(z0)


# Jordan-Wigner and Bravyi-Kitaev
tmp0 = FermionOperator('2^ 0', 3.17)
z0 = tmp0 + hermitian_conjugated(tmp0)
z1 = jordan_wigner(z0)
z2 = bravyi_kitaev(z0)
eigenspectrum(z1)
eigenspectrum(z2)


z0 = QubitOperator('X0 Y1 Z2', 88) + QubitOperator('Z1 Z4', 3.17)
z1 = reverse_jordan_wigner(z0)
z2 = jordan_wigner(z1)
z2.compress()


z0 = fermi_hubbard(x_dimension=2, y_dimension=2, tunneling=2, coulomb=1,
        chemical_potential=0.25, magnetic_field=0.5, periodic=True, spinless=True)


grid = Grid(dimensions=1, length=3, scale=1.0)
momentum_hamiltonian = jellium_model(grid, spinless=True)
position_hamiltonian = fourier_transform(momentum_hamiltonian, grid, spinless=True)


diatomic_bond_length = 0.7414
molecule = MolecularData(
    geometry=[('H', (0, 0, 0)), ('H', (0, 0, diatomic_bond_length))],
    basis='sto-3g',
    multiplicity=1,
    charge=0,
    description=str(diatomic_bond_length),
)
molecule.name
molecule.filename
molecule.n_atoms
molecule.atoms
molecule.protons
