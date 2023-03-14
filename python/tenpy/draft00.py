import numpy as np
import tenpy
import tenpy.linalg.np_conserved as npc

## Basic use of the `Array` class with trivial arrays
M = npc.Array.from_ndarray_trivial([[0., 1.], [1., 0.]])
v = npc.Array.from_ndarray_trivial([2., 4. + 1.j])
v[0] = 3
v.to_ndarray() #(np,complex128,(2,))
M_v = npc.tensordot(M, v, axes=[1, 0])
npc.inner(v.conj(), M_v, axes='range')



## Explicit definition of charges and spin-1/2 operators
# consider spin-1/2 with Sz-conservation
chinfo = npc.ChargeInfo([1])  # just a U(1) charge
# charges for up, down state
p_leg = npc.LegCharge.from_qflat(chinfo, [[1], [-1]])
Sz = npc.Array.from_ndarray([[0.5, 0.], [0., -0.5]], [p_leg, p_leg.conj()])
Sp = npc.Array.from_ndarray([[0., 1.], [0., 0.]], [p_leg, p_leg.conj()])
Sm = npc.Array.from_ndarray([[0., 0.], [1., 0.]], [p_leg, p_leg.conj()])

Hxy = 0.5 * (npc.outer(Sp, Sm) + npc.outer(Sm, Sp))
Hz = npc.outer(Sz, Sz)
H = Hxy + Hz
# here, H has 4 legs
H.iset_leg_labels(["s1", "t1", "s2", "t2"])
H = H.combine_legs([["s1", "s2"], ["t1", "t2"]], qconj=[+1, -1]) # here, H has 2 legs
H.legs[0].to_qflat().flatten() # [-2  0  0  2]
E, U = npc.eigh(H)  # diagonalize blocks individually
# E: [ 0.25 -0.75  0.25  0.25]
