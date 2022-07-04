import numpy as np
import pfapack.pfaffian

tmp0 = np.random.rand(6, 6)
A = tmp0 - tmp0.T
pfa1 = pfapack.pfaffian.pfaffian(A)
pfa2 = pfapack.pfaffian.pfaffian(A, method="H")
pfa3 = pfapack.pfaffian.pfaffian_schur(A)

print(pfa1, pfa2, pfa3)
print(pfa1**2, np.linalg.det(A))


# fail, maybe work on linux
# from pfapack.ctypes import pfaffian as cpf
# pfa1 = cpf(A)
# pfa2 = cpf(A, method="H")

# fail not apply to skew-hermitian matrix
# tmp0 = np.random.rand(6,6) + 1j*np.random.rand(6,6)
# A = tmp0 - tmp0.T.conj()
# pfapack.pfaffian.pfaffian(A)
