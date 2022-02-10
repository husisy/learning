import os
import numpy as np
import tensorflow as tf

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def next_tbd_dir(dir0='tbd00', maximum_int=100000):
    import random
    if not os.path.exists(dir0):
        os.makedirs(dir0)
    tmp1 = [x for x in os.listdir(dir0) if x[:3]=='tbd']
    exist_set = {x[3:] for x in tmp1}
    while True:
        tmp1 = str(random.randint(1,maximum_int))
        if tmp1 not in exist_set:
            break
    tbd_dir = os.path.join(dir0, 'tbd'+tmp1)
    os.mkdir(tbd_dir)
    return tbd_dir


def test_tensor_bool_conversion():
    assert tf.constant(1)==1
    assert not (tf.constant(1)!=1)
    assert tf.equal(tf.constant(1), 1)
    assert tf.constant(1)!=0
    assert bool(tf.constant(1, dtype=tf.int64))
    assert not tf.constant(0, dtype=tf.int64)


def test_tf_device_cpu_gpu():
    if len(tf.config.list_physical_devices("GPU"))>0:
        np0 = np.random.rand(3,5).astype(np.float32)
        np1 = np.random.rand(5,7).astype(np.float32)
        with tf.device('gpu:0'):
            tf_gpu = tf.linalg.matmul(tf.constant(np0), tf.constant(np1))
        with tf.device('cpu:0'):
            tf_cpu = tf.linalg.matmul(tf.constant(np0), tf.constant(np1))
        assert hfe(tf_cpu.numpy(), tf_gpu.numpy()) < 1e-5


def test_type_conversion():
    np0 = np.random.rand(3,3).astype(np.float32)
    tf0 = tf.constant(np0)
    np1 = np0 @ np0 #np.matmul
    tf1 = tf0 @ tf0 #tf.linalg.matmul
    np2 = np.matmul(tf0, tf0) #chi bao le cheng de
    tf2 = tf.linalg.matmul(np0, np0)
    assert hfe(np1, tf1.numpy()) < 1e-5
    assert hfe(np2, tf2.numpy()) < 1e-5


def test_gradient_tape(N0=3):
    np0 = np.random.rand(N0).astype(np.float32)
    tf0 = tf.constant(np0)
    with tf.GradientTape() as tape:
        tape.watch(tf0)
        tf1 = tf.math.reduce_sum(tf0**2)
    grad = tape.gradient(tf1, tf0)
    assert hfe(2*np0, grad.numpy()) < 1e-5


#TODO tensor_scatter_nd_update


def test_hessian_by_GradientTape(N0=3):
    np0 = np.random.rand(N0).astype(np.float32)
    hessian_np0 = 4 * np.prod(np0)**2 / (np0[:,np.newaxis]*np0)
    hessian_np0[np.arange(N0), np.arange(N0)] /= 2

    tf0 = tf.constant(np0)
    with tf.GradientTape(persistent=True) as tape0:
        tape0.watch(tf0)
        with tf.GradientTape() as tape1:
            tape1.watch(tf0)
            tf1 = tf.math.reduce_prod(tf0)**2
        grad_tf0 = tape1.gradient(tf1, tf0)
    hessian_tf0 = tape0.jacobian(grad_tf0, tf0, experimental_use_pfor=False) #strange
    assert hfe(hessian_np0, hessian_tf0.numpy()) < 1e-5



@tf.function
def hf0_test_function_loop_scatter(tf0, tf1):
    ret = tf.zeros(tf.shape(tf0)[1]*tf.math.reduce_sum(tf1), dtype=tf.float32)
    ind_end = tf.math.cumsum(tf1)*tf.shape(tf0)[1]
    ind_begin = tf.concat([[0],ind_end[:-1]], axis=0)
    # TODO maybe could use list comprehension
    for ind0 in tf.range(tf.shape(tf0)[0]):
        tmp0 = tf.range(ind_begin[ind0],ind_end[ind0])[:,tf.newaxis]
        tmp1 = tf.tile(tf0[ind0], (tf1[ind0],))
        ret = tf.tensor_scatter_nd_update(ret, tmp0, tmp1)
    return ret

def test_function_loop_scatter():
    np0 = np.random.rand(4, 3).astype(np.float32)
    np1 = np.random.randint(1, 4, size=(4,))
    np2 = np.concatenate([np.tile(x,y) for x,y in zip(np0,np1)], axis=0)

    tf0 = tf.constant(np0, dtype=tf.float32)
    tf1 = tf.constant(np1, dtype=tf.int32)
    tf2 = hf0_test_function_loop_scatter(tf0, tf1)
    assert hfe(np2, tf2.numpy()) < 1e-5


def test_gradient_IndexedSlices(N0=3):
    np0 = np.random.rand(3)
    np1 = np.sum(np0**2)
    grad_np0 = 2*np0
    tf0 = tf.constant(np0)
    with tf.GradientTape() as tape:
        tape.watch(tf0)
        tf1 = tf.gather(tf0*1, tf.range(N0))
        tf2 = tf.math.reduce_sum(tf1**2)
    assert hfe(grad_np0, tape.gradient(tf2, tf0).numpy()) < 1e-5
    with tf.GradientTape() as tape:
        tape.watch(tf0)
        tf1 = tf.gather(tf0, tf.range(N0))
        tf2 = tf.math.reduce_sum(tf1**2)
    #see https://www.tensorflow.org/api_docs/python/tf/GradientTape?version=stable#returns_5 IndexedSlices
    assert hfe(grad_np0, tape.gradient(tf2, tf0).values.numpy()) < 1e-5


def hf_collatz_conjecture(x):
    ret = 0
    while x!=1:
        x = (x//2) if (x%2==0) else (3*x+1)
        ret += 1
    return ret


def test_collatz_conjecture(N0=30):
    # Collatz conjecture: https://en.wikipedia.org/wiki/Collatz_conjecture
    tf_collatz_conjecture = tf.function(hf_collatz_conjecture)
    for x in range(2, N0):
        ret_ = hf_collatz_conjecture(x) #int
        ret0 = hf_collatz_conjecture(tf.constant(x, dtype=tf.dtypes.int64)) #int
        ret1 = tf_collatz_conjecture(tf.constant(x, dtype=tf.dtypes.int64)).numpy()
        assert ret_==ret0
        assert ret_==ret1


def test_complex_gradient():
    np0_real = np.random.randn(3,5).astype(np.float32)
    np0_imag = np.random.randn(3,5).astype(np.float32)
    tf0_real = tf.constant(np0_real)
    tf0_imag = tf.constant(np0_imag)
    with tf.GradientTape() as tape:
        tape.watch(tf0_real)
        tape.watch(tf0_imag)
        tf1 = tf.math.reduce_sum(tf.math.abs(tf.dtypes.complex(tf0_real, tf0_imag))**2)
    tf0_real_grad,tf0_imag_grad = tape.gradient(tf1, [tf0_real,tf0_imag])
    assert hfe(2*np0_real, tf0_real_grad.numpy()) < 1e-5
    assert hfe(2*np0_imag, tf0_imag_grad.numpy()) < 1e-5


def test_tf_model_serialization00():
    '''https://www.tensorflow.org/tutorials/keras/save_and_load'''
    hf0 = lambda: tf.keras.models.Sequential([
        tf.keras.layers.Dense(23, activation='relu', input_shape=(13,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(5),
    ])
    logdir = next_tbd_dir()
    filepath = os.path.join(logdir, 'tbd00.ckpt')

    tf0 = tf.random.uniform((3,13))
    model0 = hf0()
    ret0 = model0(tf0, training=False)
    ret1 = model0(tf0, training=False)
    assert hfe(ret0.numpy(), ret1.numpy()) < 1e-4

    model0.save_weights(filepath)
    model1 = hf0()
    model1.load_weights(filepath)
    ret2 = model1(tf0, training=False)
    assert hfe(ret0.numpy(), ret2.numpy()) < 1e-4


def test_tf_model_serialization01():
    '''https://www.tensorflow.org/tutorials/keras/save_and_load'''
    class MyModel00(tf.keras.Model):
        def __init__(self):
            super(MyModel00, self).__init__()
            self.fc0 = tf.keras.layers.Dense(23, activation='relu', input_shape=(13,))
            self.dropout0 = tf.keras.layers.Dropout(0.2)
            self.fc1 = tf.keras.layers.Dense(5)
        def call(self, x, training=False):
            x = self.fc0(x)
            x = self.dropout0(x, training=training)
            x = self.fc1(x)
            return x
    logdir = next_tbd_dir()
    filepath = os.path.join(logdir, 'tbd00.ckpt')

    tf0 = tf.random.uniform((3,13))
    model0 = MyModel00()
    ret0 = model0(tf0, training=False)
    ret1 = model0(tf0, training=False)
    assert hfe(ret0.numpy(), ret1.numpy()) < 1e-4

    model0.save_weights(filepath)
    model1 = hf0()
    model1.load_weights(filepath)
    ret2 = model1(tf0, training=False)
    assert hfe(ret0.numpy(), ret2.numpy()) < 1e-4
