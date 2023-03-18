import numpy as np

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def rand_unitary_matrix(N0, tag_complex=True, seed=None):
    np_rng = np.random.default_rng(seed)
    if tag_complex:
        tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T.conj()
    else:
        tmp0 = np_rng.normal(size=(N0,N0))
        tmp0 = tmp0 + tmp0.T
    ret = np.linalg.eigh(tmp0)[1]
    return ret


def rand_hermite_matrix(N0, min_eig=1, max_eig=2, tag_complex=True, seed=None):
    np_rng = np.random.default_rng(seed)
    if (min_eig is None) and (max_eig is None):
        if tag_complex:
            tmp0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
            ret = tmp0 + tmp0.T.conj()
        else:
            tmp0 = np_rng.normal(size=(N0,N0))
            ret = tmp0 + tmp0.T
    else:
        assert (min_eig is not None) and (max_eig is not None)
        EVL = np_rng.uniform(min_eig, max_eig, size=(N0,))
        EVC = rand_unitary_matrix(N0, tag_complex, seed=np_rng.integers(10000))
        tmp0 = EVC.T.conj() if tag_complex else EVC.T
        ret = (EVC * EVL) @ tmp0
    return ret


def rand_invertible_matrix(N0, min_svd=1, max_svd=2, tag_complex=True, seed=None):
    assert (max_svd>=min_svd) and (min_svd>0)
    np_rng = np.random.default_rng(seed)
    unitary1 = rand_unitary_matrix(N0, tag_complex, seed=np_rng.integers(10000))
    unitary2 = rand_unitary_matrix(N0, tag_complex, seed=np_rng.integers(10000))
    sign = 2*(np_rng.integers(2, size=N0)>=1) - 1
    diag = np_rng.uniform(min_svd,max_svd,size=N0)*sign
    ret = (unitary1 * diag) @ unitary2
    return ret


def rand_commute_matrix(N0, kind='hermitian', tag_complex=True, seed=None):
    assert kind in {'hermitian','normal','any'}
    np_rng = np.random.default_rng(seed)
    if kind=='hermitian':
        unitary0 = rand_unitary_matrix(N0, tag_complex, seed=np_rng.integers(10000))
        unitary1 = unitary0.T.conj()
        np1 = np_rng.normal(size=N0)
        np2 = np_rng.normal(size=N0)
        ret0 = (unitary0 * np1) @ unitary1
        ret1 = (unitary0 * np2) @ unitary1
    elif kind=='normal':
        unitary0 = rand_unitary_matrix(N0, tag_complex, seed=np_rng.integers(10000))
        unitary1 = unitary0.T.conj()
        np1 = np_rng.normal(size=N0) + 1j*np_rng.normal(size=N0)
        np2 = np_rng.normal(size=N0) + 1j*np_rng.normal(size=N0)
        ret0 = (unitary0 * np1) @ unitary1
        ret1 = (unitary0 * np2) @ unitary1
    else:
        np0 = rand_invertible_matrix(N0, min_svd=1, max_svd=2)
        np0_inv = np.linalg.inv(np0)
        np1 = np_rng.normal(size=N0) + 1j*np_rng.normal(size=N0)
        np2 = np_rng.normal(size=N0) + 1j*np_rng.normal(size=N0)
        ret0 = (np0 * np1) @ np0_inv
        ret1 = (np0 * np2) @ np0_inv
    return ret0,ret1


def rand_invariant_matrix(N0, kind='normal', tag_complex=True, seed=None):
    # UAU^H = A
    assert kind in {'hermitian','normal','unitary'}
    np_rng = np.random.default_rng(seed)
    unitary0 = rand_unitary_matrix(N0, tag_complex, seed=np_rng.integers(10000))
    if kind=='normal':
        np0 = np_rng.normal(size=N0) + 1j*np_rng.uniform(0, 2*np.pi, size=N0)
    elif kind=='hermitian':
        np0 = np_rng.normal(size=N0)
    else:
        np0 = 1j*np_rng.uniform(0, 2*np.pi, size=N0)
    tmp0 = np.exp(1j*np_rng.uniform(0, 2*np.pi, size=N0))
    ret0 = (unitary0*tmp0) @ unitary0.T.conj()
    ret1 = (unitary0*np.exp(np0)) @ unitary0.T.conj()
    return ret0,ret1


def test_rand_commute_matrix():
    N0 = 5
    for kind in ['hermitian','normal','any']:
        for tag_complex in [True,False]:
            z0,z1 = rand_commute_matrix(N0, kind, tag_complex)
            assert hfe(z0@z1, z1@z0) < 1e-10


def test_rand_invariant_matrix():
    N0 = 5
    for kind in ['hermitian','normal','unitary']:
        for tag_complex in [True,False]:
            z0,z1 = rand_invariant_matrix(N0, kind, tag_complex)
            assert hfe((z0 @ z1) @ z0.T.conj(), z1) < 1e-10


def test_rand_unitary_matrix():
    N0 = 5
    tmp0 = rand_unitary_matrix(N0, tag_complex=True)
    assert hfe(tmp0 @ tmp0.T.conj(), np.eye(N0)) < 1e-5
    tmp0 = rand_unitary_matrix(N0, tag_complex=False)
    assert hfe(tmp0 @ tmp0.T.conj(), np.eye(N0)) < 1e-5


def test_rand_hermite_matrix():
    N0 = 5
    tmp0 = rand_hermite_matrix(N0, tag_complex=True)
    assert hfe(tmp0,tmp0.T.conj())<1e-7
    tmp0 = rand_hermite_matrix(N0, tag_complex=False)
    assert hfe(tmp0,tmp0.T)<1e-7
    tmp0 = rand_hermite_matrix(N0, min_eig=None, max_eig=None, tag_complex=True)
    assert hfe(tmp0,tmp0.T.conj())<1e-7
    tmp0 = rand_hermite_matrix(N0, min_eig=None, max_eig=None, tag_complex=False)
    assert hfe(tmp0,tmp0.T)<1e-7


def rotate_vec0_to_vec1(vec0, vec1):
    N0 = vec0.shape[0]
    vec0 = vec0 / np.linalg.norm(vec0)
    vec1 = vec1 / np.linalg.norm(vec1)
    np0 = vec1 - vec0
    tmp0 = np.linalg.norm(np0)
    assert tmp0>1e-7
    np0 /= tmp0
    # https://en.wikipedia.org/wiki/Householder_transformation
    ret = np.eye(N0) - 2*np0[:,np.newaxis]*np0.conj()
    return ret

# def _rand_stochastic_matrix_hf0(N0, np_rng):
#     tmp0 = np_rng.normal(size=(N0-1,N0-1))
#     tmp1 = np.eye(N0, dtype=np.float64)
#     tmp1[:-1,:-1] = np.linalg.eigh(tmp0 + tmp0.T)[1]
#     ret = tmp1
#     return ret

# def rand_doubly_stochastic_matrix(N0, kind='right', seed=None):
#     assert kind in {'left','right','doubly'}
#     np_rng = np.random.default_rng(seed)

#     tmp0 = np.zeros(N0)
#     tmp0[-1] = 1
#     householder_rotate = rotate_vec0_to_vec1(tmp0, np.ones(N0))
#     EVC0 = _rand_stochastic_matrix_hf0(N0, np_rng)
#     EVC1 =  _rand_stochastic_matrix_hf0(N0, np_rng)
#     EVL = np.concatenate([np_rng.uniform(0, 1, size=N0-1), [1]])
#     ret = householder_rotate @ (EVC0*EVL) @ EVC1 @ householder_rotate
#     # fail, cannot guarantee positive
#     return ret
