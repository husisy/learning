import numpy as np
import scipy.linalg

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))


def tf_hessian_matrix(hf0, x0):
    import tensorflow as tf
    tf0 = tf.convert_to_tensor(x0)
    with tf.GradientTape() as tape0:
        tape0.watch(tf0)
        with tf.GradientTape() as tape1:
            tape1.watch(tf0)
            tf1 = hf0(tf0)
        grad_tf0 = tape1.gradient(tf1, tf0)
    hessian_tf0 = tape0.jacobian(grad_tf0, tf0)
    return hessian_tf0.numpy()

def np_hessian_matrix(hf0, x0, zero_eps=1e-4):
    # float32 precision 10-7
    num0 = x0.shape[0]
    ret = np.zeros((num0,num0), dtype=x0.dtype)
    for ind0 in range(num0):
        for ind1 in range(ind0, num0):
            tmp0 = x0.copy()
            tmp0[ind0] += zero_eps
            tmp0[ind1] += zero_eps

            tmp1 = x0.copy()
            tmp1[ind0] -= zero_eps
            tmp1[ind1] -= zero_eps

            tmp2 = x0.copy()
            tmp2[ind0] += zero_eps
            tmp2[ind1] -= zero_eps

            tmp3 = x0.copy()
            tmp3[ind0] -= zero_eps
            tmp3[ind1] += zero_eps

            tmp4 = (hf0(tmp0) + hf0(tmp1) - hf0(tmp2) - hf0(tmp3)) / (4*zero_eps*zero_eps)
            ret[ind0,ind1] = tmp4
            ret[ind1,ind0] = tmp4
    return ret

def test_hessian_matrix():
    hf0 = lambda x: (x[0]**(1/2)) * (x[1]**(1/3)) * (x[2]**(1/4)) * (x[3]**(1/5))
    x0 = np.random.rand(4) + 1
    ret_ = tf_hessian_matrix(hf0, x0)
    ret = np_hessian_matrix(hf0, x0, zero_eps=1e-4)
    assert hfe(ret_, ret) < 1e-6


def test_np_choose(N0=1000, N1=10):
    # TODO move to indexing
    np0 = np.random.rand(N0,N1)
    np1 = np.random.randint(0,N1,N0)

    ret_ = np0[np.arange(N0),np1]
    ret = np.choose(np1, np0.transpose((1,0)))
    assert hfe(ret_, ret) < 1e-7


def pauli_rotation(a, theta, phi):
    PauliX = np.array([[0,1],[1,0]])
    PauliY = np.array([[0,-1j],[1j,0]])
    PauliZ = np.array([[1,0],[0,-1]])
    tmp0 = np.sin(theta)*np.cos(phi)*PauliX
    tmp1 = np.sin(theta)*np.sin(phi)*PauliY
    tmp2 = np.cos(theta)*PauliZ
    ret = scipy.linalg.expm(1j*a * (tmp0 + tmp1 + tmp2))
    return ret

def pauli_rotation_euler(a, theta, phi):
    ca = np.cos(a)
    sa = np.sin(a)
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    tmp0 = ca + 1j*sa*ct
    tmp3 = ca - 1j*sa*ct
    tmp1 = sa*st*(sp + 1j*cp)
    tmp2 = sa*st*(-sp + 1j*cp)
    ret = np.array([[tmp0,tmp1],[tmp2,tmp3]])
    return ret

def test_pauli_rotation():
    # see https://en.wikipedia.org/wiki/Pauli_matrices#Exponential_of_a_Pauli_vector
    a = np.random.uniform(0, 2*np.pi)
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, 2*np.pi)
    ret_ = pauli_rotation(a, theta, phi)
    ret = pauli_rotation_euler(a, theta, phi)
    assert hfe(ret_, ret) < 1e-7


def test_covariance(N0=3, N1=23):
    np0 = np.random.randn(N0, N1)
    ret_ = np.cov(np0)
    tmp0 = np0 - np0.mean(1, keepdims=True)
    ret0 = (tmp0 @ tmp0.T) / (N1-1)
    assert hfe(ret_, ret0) < 1e-7
