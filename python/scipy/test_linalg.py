import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfH = lambda x: np.conjugate(x.T)
hf_randc = lambda *size: np.random.randn(*size) + 1j*np.random.randn(*size)
hf_hermite = lambda x: (x + np.conjugate(x.T))/2

def test_solve_banded(N0=233, num_l=3, num_u=5):
    ab = np.random.randn(num_l+num_u+1, N0)
    ab[num_u+1] = ab[num_u+1] + N0/2 #avoid irreversible
    b = np.random.rand(N0)
    for ind0 in range(num_l):
        ab[num_u+1+ind0,(N0-ind0-1):] = 0
    for ind0 in range(num_u):
        ab[ind0,:(num_u-ind0)] = 0
    ret = scipy.linalg.solve_banded((num_l,num_u), ab, b)

    A = scipy.sparse.spdiags(ab, list(range(num_u,-num_l-1,-1)), N0, N0).todense()
    ret_ = np.linalg.solve(A, b)
    assert hfe(ret_, ret) < 1e-7


def test_tridiagonal_solve_banded(N0=233):
    middle_element = np.random.randn(N0) + N0/2
    upper_element = np.random.randn(N0-1)
    lower_element = np.random.randn(N0-1)
    b = np.random.randn(N0)
    ab = np.block([[np.array([0]), upper_element], [middle_element], [lower_element, np.array([0])]])
    ret = scipy.linalg.solve_banded((1,1), ab, b)
    A = np.diag(middle_element) + np.diag(upper_element,1) + np.diag(lower_element,-1)
    ret_ = np.linalg.solve(A, b)
    assert hfe(ret_, ret) < 1e-7


def matrix_to_diagonal(np0, offset, return_array=True):
    assert np0.ndim==2
    ret = [np.diagonal(np0,x) for x in offset]
    if return_array:
        N0 = np0.shape[1]
        hf0 = lambda x,z: np.concatenate([[0]*z,x,[0]*(N0-z-len(x))])
        ret = np.stack([hf0(x,max(0,y)) for x,y in zip(ret,offset)])
    return ret


def test_matrix_to_diagonal(N0=5, N1=7, offset=(-1,1,0,2)):
    np0 = np.random.randn(N0, N1)
    diagonal_list = matrix_to_diagonal(np0, offset, return_array=False)
    z0 = scipy.sparse.diags(diagonal_list, offset, shape=(N0,N1)).todense()
    diagonal_array = matrix_to_diagonal(np0, offset, return_array=True)
    z1 = scipy.sparse.spdiags(diagonal_array, offset, N0, N1).todense()
    assert hfe(z0,z1) < 1e-7


def test_eigh_tridiagonal(N0=13):
    diagonal_element = np.random.randn(N0)
    upper_element = np.random.randn(N0-1)
    EVL,EVC = scipy.linalg.eigh_tridiagonal(diagonal_element, upper_element)
    tmp0 = np.diag(diagonal_element) + np.diag(upper_element,1) + np.diag(upper_element,-1)
    EVL_,EVC_ = np.linalg.eigh(tmp0)
    assert hfe(np.diag(EVL_), EVC_.T @ tmp0 @ EVC_) < 1e-7
    assert hfe(np.diag(EVL), EVC.T @ tmp0 @ EVC) < 1e-7
    assert hfe(EVL_, EVL) < 1e-7


def generate_hermite_matrix(N0, min_eig=1, max_eig=2):
    ret = np.random.randn(N0,N0) + 1j*np.random.randn(N0,N0)
    ret = ret + np.conjugate(ret.T)
    eig0 = scipy.sparse.linalg.eigs(ret, k=1, which='SR', return_eigenvectors=False)
    eig1 = scipy.sparse.linalg.eigs(ret, k=1, which='LR', return_eigenvectors=False)
    ret = (ret - eig0*np.eye(N0)) * (max_eig-min_eig)/(eig1-eig0) + min_eig*np.eye(N0)
    return ret


def test_np_cholesky(N0=23):
    np0 = generate_hermite_matrix(N0, min_eig=1, max_eig=2)
    L = np.linalg.cholesky(np0)
    assert hfe(L, np.tril(L)) < 1e-7
    assert hfe(np0, L @ np.conjugate(L.T)) < 1e-7


def generate_unitary_matrix(N0, tag_real=False):
    if tag_real:
        tmp0 = np.random.randn(N0, N0)
        _,ret = np.linalg.eig(tmp0 + tmp0.T)
    else:
        tmp0 = np.random.randn(N0, N0) + 1j*np.random.rand(N0,N0)
        _,ret = np.linalg.eig(tmp0 + np.conjugate(tmp0.T))
    return ret


def test_sp_eig(N0=233):
    tmp0 = np.random.randn(N0,N0) + 1j*np.random.randn(N0,N0)
    np0 = tmp0 + np.conjugate(tmp0.T)
    eigen_value,vec_left,vec_right = scipy.linalg.eig(np0, left=True, right=True)
    assert hfe(hfH(vec_left)@vec_left, np.eye(N0)) < 1e-7
    assert hfe(hfH(vec_right)@vec_right, np.eye(N0)) < 1e-7
    assert hfe(np0 @ vec_right, vec_right*eigen_value) < 1e-7
    assert hfe(hfH(vec_left)@np0, hfH(vec_left)*eigen_value[:,np.newaxis]) < 1e-7


def test_qr():
    for N0,N1 in [(5,7),(7,5)]:
        np0 = np.random.randn(N0, N1)
        npq, npr = np.linalg.qr(np0)
        assert hfe(npq.T.conj() @ npq, np.eye(min(N0,N1))) < 1e-7
        assert hfe(npr, np.triu(npr)) < 1e-7
        assert hfe(np0, npq@npr) < 1e-7

def test_generalized_eigh():
    N0 = 5
    np_rng = np.random.default_rng()
    hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)
    np0 = hf_hermite(hf_randc(N0, N0))
    tmp0 = hf_randc(N0, N0)
    np1 = tmp0 @ tmp0.T.conj() + np.eye(N0)/100 #SPD
    EVL,EVC = scipy.linalg.eigh(np0, b=np1)
    assert np.abs(np0 @ EVC - np1 @ (EVC*EVL)).max() < 1e-7 #A V = B V lambda
    assert hfe(EVC.T.conj() @ np1 @ EVC, np.eye(N0)) < 1e-7 #V^T B V = I


def test_polar():
    N0 = 5
    np_rng = np.random.default_rng()
    hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)
    npA = hf_randc(N0,N0)
    npU,npP = scipy.linalg.polar(npA, side='right') #default right
    assert np.abs(npA - npU @ npP).max() < 1e-7
    assert hfe(npU @ npU.T.conj(), np.eye(N0)) < 1e-7
    assert np.all(np.linalg.eigvalsh(npP)>=0)


def test_schur():
    # https://en.wikipedia.org/wiki/Schur_decomposition
    N0 = 5
    np0 = hf_randc(N0, N0)
    Tmat, Umat = scipy.linalg.schur(np0, output='complex')
    assert np.abs(np0 - Umat @ Tmat @ Umat.T.conj()).max() < 1e-7
    assert hfe(Umat @ Umat.T.conj(), np.eye(N0))<1e-7
    assert np.abs(np.tril(Tmat, -1)).max() < 1e-7


def test_solve_sylvester():
    N0 = 4
    np_rng = np.random.default_rng()
    np0 = np_rng.normal(size=(N0,N0))
    np1 = np_rng.normal(size=(N0,N0))
    np2 = np_rng.normal(size=(N0,N0))
    ret0 = scipy.linalg.solve_sylvester(np0, np1, np2)
    assert np.abs(np0@ret0 + ret0@np1 - np2).max() < 1e-7
    ret_ = np.linalg.solve(np.kron(np0, np.eye(N0)) + np.kron(np.eye(N0),np1.T), np2.reshape(-1)).reshape(N0,N0)
    assert np.abs(ret0-ret_).max()<1e-7
