import numpy as np

import quaternionic

np_rng = np.random.default_rng()

np0 = np_rng.normal(size=(17,11,4))
qt0 = quaternionic.array(np0)


qt1 = quaternionic.array([1.2, 2.3, 3.4, 4.5]) #w x y z
qt1.w
qt1.x #qt1.i
qt1.y #qt1.j
qt1.z #qt.k
qt1.scalar #qt1.w qt1.real
qt1.vector #qt1.imag
