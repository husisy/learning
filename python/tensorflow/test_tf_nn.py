import numpy as np
import tensorflow as tf

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_tf_nn_dropout(drpr=0.233):
    np0 = np.random.rand(3, 5) + 1
    tf1 = tf.nn.dropout(np0, drpr)
    ind0 = tf1.numpy() < 1e-5
    np1 = np0.copy() / (1-drpr)
    np1[ind0] = 0
    assert hfe(np1, tf1.numpy()) < 1e-5


def test_tf_nn_sigmoid_cross_entropy_with_logits(N0=233):
    np0 = np.random.randint(0, 2, N0)
    np1 = np.random.randn(N0).astype(np.float32)
    ret_ = np.zeros(N0, dtype=np.float32)
    tmp0 = 1 / (1 + np.exp(-np1))
    ret_[np0==0] = - np.log(1- tmp0[np0==0])
    ret_[np0==1] = - np.log(tmp0[np0==1])
    ret = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.constant(np0, dtype=tf.float32), logits=tf.constant(np1))
    assert hfe(ret_, ret.numpy()) < 1e-5
