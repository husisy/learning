import numpy as np

import clifford


## 2-dimensional clifford algebra (G2)
layout, blades = clifford.Cl(2)
e1 = blades['e1']
e2 = blades['e2']
e12 = blades['e12']

e1*e2  # geometric product
e1|e2  # inner product
e1^e2  # outer product

# reflection
a = e1+e2     # the vector
n = e1        # the reflector
-n*a*n.inv()  # reflect `a` in hyperplane normal to `n`

# rotation
R = np.e**(np.pi/4*e12)  # enacts rotation by pi/2
R*e1*~R    # rotate e1 by pi/2 in the e12-plane


## 3-dimension clifford algebra (G3)
layout, blades = clifford.Cl(3)
e1 = blades['e1']
e2 = blades['e2']
e3 = blades['e3']
e12 = blades['e12']
e13 = blades['e13']
e23 = blades['e23']
e123 = blades['e123']

e1*e2  # geometric product
e1|e2  # inner product
e1^e2  # outer product
