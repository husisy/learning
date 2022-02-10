import numpy as np
import pyscf
import pyscf.dft

mol_h2o = pyscf.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='6-31g')

## Uniform grids https://github.com/pyscf/pyscf/blob/master/examples/dft/30-ao_value_on_grid.py
N0 = 20
tmp0 = np.linspace(-10,10,N0)
coords = np.stack([
    np.broadcast_to(tmp0[np.newaxis,np.newaxis,:], (N0,N0,N0)).reshape(-1),
    np.broadcast_to(tmp0[np.newaxis,:,np.newaxis], (N0,N0,N0)).reshape(-1),
    np.broadcast_to(tmp0[:,np.newaxis,np.newaxis], (N0,N0,N0)).reshape(-1),
], axis=1)

ao_value = pyscf.dft.numint.eval_ao(mol_h2o, coords) #(np,float64,(8000,13))
ao_value = pyscf.dft.numint.eval_ao(mol_h2o, coords, deriv=1) #(np,float64,(4,8000,13))


## grid scheme https://github.com/pyscf/pyscf/blob/master/examples/dft/11-grid_scheme.py
method = pyscf.dft.RKS(mol_h2o) #DFT(LDA)
method.kernel() #-75.81792888333536

# See pyscf/dft/radi.py for more radial grid schemes
method = pyscf.dft.RKS(mol_h2o)
method.grids.radi_method = pyscf.dft.mura_knowles #pyscf.dft.gauss_chebeshev, pyscf.dft.delley
method.kernel() #-75.81792884913116

# See pyscf/dft/gen_grid.py for detail of the grid weight scheme, Stratmann-Scuseria weight scheme
method = pyscf.dft.RKS(mol_h2o)
method.grids.becke_scheme = pyscf.dft.stratmann #pyscf.dft.original_becke
method.kernel() #-75.81792885735909

method = pyscf.dft.RKS(mol_h2o)
method.grids.level = 4 #Grids level 0 - 9.  Big number indicates dense grids. Default is 3
method.kernel() #-75.8179288594003

# Specify mesh grid for certain atom
method = pyscf.dft.RKS(mol_h2o)
method.grids.atom_grid = {'O': (100, 770)}
method.kernel() #-75.81792888767139

# Specify mesh grid for all atoms
method = pyscf.dft.RKS(mol_h2o)
method.grids.atom_grid = (100, 770)
method.kernel() #-75.81792884078422

# Disable pruning grids near core region
method = pyscf.dft.RKS(mol_h2o)
method.grids.prune = None #pyscf.dft.sg1_prune
method.kernel() #-75.81792888578866
