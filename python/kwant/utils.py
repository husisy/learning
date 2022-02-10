import types
import numpy as np

s0 = np.array([[1, 0], [0, 1]])
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
pauli = types.SimpleNamespace(
    s0 = s0,
    sx = sx,
    sy = sy,
    sz = sz,
    tau_x = sx,
    tau_y = sy,
    tau_z = sz,
    s0s0 = np.kron(s0, s0),
    s0sx = np.kron(s0, sx),
    s0sy = np.kron(s0, sy),
    s0sz = np.kron(s0, sz),
    sxs0 = np.kron(sx, s0),
    sxsx = np.kron(sx, sx),
    sxsy = np.kron(sx, sy),
    sxsz = np.kron(sx, sz),
    sys0 = np.kron(sy, s0),
    sysx = np.kron(sy, sx),
    sysy = np.kron(sy, sy),
    sysz = np.kron(sy, sz),
    szs0 = np.kron(sz, s0),
    szsx = np.kron(sz, sx),
    szsy = np.kron(sz, sy),
    szsz = np.kron(sz, sz),
)
