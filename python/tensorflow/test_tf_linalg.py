import numpy as np
import scipy.linalg
import tensorflow as tf

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_randc = lambda *size: np.random.randn(*size) + 1j*np.random.randn(*size)
hf_hermite = lambda x: (x + np.conjugate(x.T))/2

def test_tf_linalg_matmul(N0=3, N1=4, N2=5):
    np1 = np.random.rand(N0, N1)
    np2 = np.random.rand(N1, N2)
    np3 = np.matmul(np1, np2)
    tf3 = tf.linalg.matmul(tf.convert_to_tensor(np1), tf.convert_to_tensor(np2))
    assert hfe(np3, tf3.numpy()) < 1e-7


def test_tf_linalg_expm_gradient(N0=3):
    np0 = hf_hermite(hf_randc(N0, N0))
    np1 = np.random.rand(N0, N0)
    zero_eps = 1e-5

    ret_ = np.zeros((N0,N0), dtype=np.complex)
    hf0 = lambda x: np.sum(np.abs(scipy.linalg.expm(x)) * np1)
    for ind0 in range(N0):
        for ind1 in range(N0):
            tmp0 = np0.copy()
            tmp0[ind0,ind1] = tmp0[ind0,ind1] + zero_eps
            tmp0 = hf0(tmp0)

            tmp1 = np0.copy()
            tmp1[ind0,ind1] = tmp1[ind0,ind1] - zero_eps
            tmp1 = hf0(tmp1)

            tmp2 = np0.copy()
            tmp2[ind0,ind1] = tmp2[ind0,ind1] + zero_eps*1j
            tmp2 = hf0(tmp2)

            tmp3 = np0.copy()
            tmp3[ind0,ind1] = tmp3[ind0,ind1] - zero_eps*1j
            tmp3 = hf0(tmp3)

            ret_[ind0,ind1] = (tmp0-tmp1)/(2*zero_eps) + 1j*(tmp2-tmp3)/(2*zero_eps)

    with tf.device('cpu:0'): #@20200705 tf.linalg.expm() fail on windows, pass on ubuntu-18.04
        tf0 = tf.constant(np0.real)
        tf1 = tf.constant(np0.imag)
        tf2 = tf.constant(np1)
        with tf.GradientTape() as tape:
            tape.watch(tf0)
            tape.watch(tf1)
            tf3 = tf.abs(tf.linalg.expm(tf.dtypes.complex(tf0, tf1)))
        tmp0,tmp1 = tape.gradient(tf3, [tf0,tf1], tf2)
    ret = tmp0.numpy() + tmp1.numpy()*1j
    assert hfe(ret_, ret) < 1e-7
