import numpy as np
import tensorflow as tf

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_tf_signal_fft(N0=3, N1=5):
    np0 = (np.random.randn(N0, N1) + np.random.randn(N0,N1)*1j).astype(np.complex64)
    tf0 = tf.constant(np0, dtype=tf.complex64)

    tf1 = tf.signal.fft(tf0)
    np1 = np.fft.fft(np0)
    assert hfe(np1, tf1.numpy()) < 1e-5

    tf1 = tf.signal.fft2d(tf0)
    np1 = np.fft.fft2(np0)
    assert hfe(np1, tf1.numpy()) < 1e-5


def test_tf_signal_ifft(N0=3, N1=5):
    np0 = (np.random.randn(N0, N1) + np.random.randn(N0,N1)*1j).astype(np.complex64)
    tf0 = tf.constant(np0, dtype=tf.complex64)

    tf1 = tf.signal.ifft(tf0)
    np1 = np.fft.ifft(np0)
    assert hfe(np1, tf1.numpy()) < 1e-5

    tf1 = tf.signal.ifft2d(tf0)
    np1 = np.fft.ifft2(np0)
    assert hfe(np1, tf1.numpy()) < 1e-5


def tf_fftshift(x, axis):
    num0 = (x.shape[axis]+1)//2
    tmp0 = (slice(None),)*axis + (slice(num0,None),)
    tmp1 = (slice(None),)*axis + (slice(None,num0),)
    ret = tf.concat([x[tmp0], x[tmp1]], axis=axis)
    return ret

def tf_ifftshift(x, axis):
    num0 = (x.shape[axis])//2
    tmp0 = (slice(None),)*axis + (slice(num0,None),)
    tmp1 = (slice(None),)*axis + (slice(None,num0),)
    ret = tf.concat([x[tmp0], x[tmp1]], axis=axis)
    return ret

def test_tf_fftshift_ifftshift():
    np0 = np.random.randn(5,8)
    tf0 = tf.constant(np0, dtype=tf.float32)
    tf1 = tf_fftshift(tf0, 0)
    np1 = np.fft.fftshift(np0, 0)
    assert hfe(np1, tf1.numpy()) < 1e-5
    tf1 = tf_fftshift(tf0, 1)
    np1 = np.fft.fftshift(np0, 1)
    assert hfe(np1, tf1.numpy()) < 1e-5
    tf1 = tf_ifftshift(tf0, 0)
    np1 = np.fft.ifftshift(np0, 0)
    assert hfe(np1, tf1.numpy()) < 1e-5
    tf1 = tf_ifftshift(tf0, 1)
    np1 = np.fft.ifftshift(np0, 1)
    assert hfe(np1, tf1.numpy()) < 1e-5
