import os
import numpy as np

import ase
import gpaw

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())


# basic
mol_h2o = ase.Atoms('H2', positions=[(0,0,0),(0,0,0.74)], cell=(6,6,6))
mol_h2o.center()
mol_h2o.calc = gpaw.GPAW(nbands=2, txt=hf_file('h2.txt'), xc='LDA', gpts=(32,32,32))
# xc='PBE'
force = mol_h2o.get_forces()
# 2-electron, LDA, spin-paired calculation, (32,32,32) grid
