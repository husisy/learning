import numpy as np
import cvxpy

## SDP backward
hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)
hf_hermite = lambda x: (x+x.T.conj())/2

np_rng = np.random.default_rng(3)

def random_density_matrix(N, np_rng, use_complex=True):
    tmp0 = np_rng.normal(size=(N,N))
    if use_complex:
        tmp0 = tmp0 + 1j* np_rng.normal(size=(N,N))
    U = np.linalg.svd(tmp0)[0]
    tmp0 = np_rng.uniform(0, 1, size=N)
    rho = (U * (tmp0/tmp0.sum())) @ U.T.conj() # SDP, trace 1
    return rho

def _real_sdp_backward(rho0_np, rho1_np, use_complex, use_backward):
    N = rho0_np.shape[0]
    rho0 = cvxpy.Parameter((N,N), complex=use_complex)
    rho0.value = rho0_np
    op = cvxpy.Variable((N,N), complex=use_complex)
    obj = cvxpy.Maximize(cvxpy.real(cvxpy.trace(op@rho0)))
    constraints = [op>>0, rho1_np>>op]
    prob = cvxpy.Problem(obj, constraints)
    if use_backward:
        prob.solve(requires_grad=True, eps=1e-10)
        op.gradient = rho0_np.T
        prob.backward()
        ret = prob.value, rho0.gradient.copy(), op.value.copy()
    else:
        prob.solve()
        ret = prob.value, op.value.copy()
    return ret

# test_real_sdp_backward()
np_rng = np.random.default_rng(233)
N = 4
use_complex = False

rho0_np = random_density_matrix(N, np_rng, complex=False) - np.eye(N)/N #subtract identity to make this problem not too trivial
rho1_np = random_density_matrix(N, np_rng, complex=True) #error if use_complex=False

_,ret0,op_np = _real_sdp_backward(rho0_np, rho1_np, use_complex=False, use_backward=True)

# TODO NOT equal to the finite difference solution
zero_eps = 1e-3
ret_ = np.zeros_like(ret0)
for ind0,ind1 in ((x,y) for x in range(N) for y in range(x,N)):
    tmp0,tmp1 = [rho0_np.copy() for _ in range(2)]
    if ind0==ind1:
        tmp0[ind0,ind0] += zero_eps
        tmp1[ind0,ind0] -= zero_eps
        tmp0 = _real_sdp_backward(tmp0, rho1_np, use_complex=False, use_backward=False)[0]
        tmp1 = _real_sdp_backward(tmp1, rho1_np, use_complex=False, use_backward=False)[0]
        ret_[ind0,ind0] = (tmp0-tmp1)/(2*zero_eps)
    else:
        tmp0[ind0,ind1] += zero_eps
        tmp0[ind1,ind0] += zero_eps
        tmp1[ind0,ind1] -= zero_eps
        tmp1[ind1,ind0] -= zero_eps
        tmp0 = _real_sdp_backward(tmp0, rho1_np, use_complex=False, use_backward=False)[0]
        tmp1 = _real_sdp_backward(tmp1, rho1_np, use_complex=False, use_backward=False)[0]
        ret_[ind0,ind1] = (tmp0-tmp1)/(2*zero_eps)
        ret_[ind1,ind0] = (tmp0-tmp1)/(2*zero_eps)




# rho0 = cvxpy.Parameter((N,N), complex=use_complex)
# rho0.value = rho0_np
# op = cvxpy.Variable((N,N), complex=use_complex)
# obj = cvxpy.Maximize(cvxpy.real(cvxpy.trace(op@rho0)))
# constraints = [op>>0, rho1_np>>op]
# prob = cvxpy.Problem(obj, constraints)
# prob.solve(requires_grad=True)
# print(prob.value) #0.03440711028904186 if complex, otherwise 0.07112770263050003

# assert prob.is_dcp(dpp=True)
# # assert prob.is_dgp(dpp=True)==False

# prob.backward() #fail with AttributeError: 'NoneType' object has no attribute 'split_adjoint'
# print(rho.gradient)




import numpy as np
import cvxpy

# https://math.stackexchange.com/a/2012242
hf_complex_to_real = lambda x: np.block([[x.real,-x.imag],[x.imag,x.real]])
hf_real_to_complex = lambda x: x[:(x.shape[0]//2),:(x.shape[1]//2)] + 1j*x[(x.shape[0]//2):,:(x.shape[1]//2)]

def random_density_matrix(N, np_rng, complex=True):
    tmp0 = np_rng.normal(size=(N,N))
    if complex:
        tmp0 = tmp0 + 1j* np_rng.normal(size=(N,N))
    U = np.linalg.svd(tmp0)[0]
    tmp0 = np_rng.uniform(0, 1, size=N)
    rho = (U * (tmp0/tmp0.sum())) @ U.T.conj() # SDP, trace 1
    return rho

np_rng = np.random.default_rng(233)
N = 4
rho_np = random_density_matrix(N, np_rng, complex=True) - np.eye(N)/N
rho1_np = random_density_matrix(N, np_rng)

rho = cvxpy.Parameter((2*N,2*N))
rho.value = hf_complex_to_real(rho_np)
op = cvxpy.Variable((2*N,2*N))
obj = cvxpy.Maximize(cvxpy.trace(op@rho)/2)
rho1 = rho1_np
constraints = [
    op>>0,
    op[:N,:N]==op[N:,N:],
    op[N:,:N]==-op[:N,N:],
    hf_complex_to_real(rho1_np)>>op,
]
prob = cvxpy.Problem(obj, constraints)
prob.solve(requires_grad=True)
print(prob.value) #0.03440711028904186

assert prob.is_dcp(dpp=True)

prob.backward()
print(hf_real_to_complex(rho.gradient))
