import numpy as np
import tensorflow as tf

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_tf_keras_layers_SimpleRNNCell(N0=3, N1=5, N2=7):
    rnn_cell = tf.keras.layers.SimpleRNNCell(N2)
    np_x = np.random.rand(N0, N1).astype(np.float32)
    np_state = [np.random.rand(N0,N2).astype(np.float32)]

    tf_output,tf_output_state = rnn_cell(tf.constant(np_x), [tf.constant(np_state[0])])
    np_xkernel,np_statekernel,np_bias = rnn_cell.get_weights()
    ret_ = np.tanh(np.matmul(np_x, np_xkernel) + np.matmul(np_state[0], np_statekernel) + np_bias)
    assert hfe(ret_, tf_output.numpy()) < 1e-5
    assert hfe(ret_, tf_output_state[0].numpy()) < 1e-5

    # TODO export graph
    # from utils import next_tbd_dir
    # @tf.function
    # def rnn_fn(x, state):
    #     return rnn_cell(x, state)
    # logdir = next_tbd_dir()
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True)
    # rnn_fn(tf_x, tf_state)
    # with writer.as_default():
    #     tf.summary.trace_export(name='rnn_cell', step=0)


def test_tf_keras_layers_Dense(N0=3, N1=5, N2=7):
    np_x = np.random.randn(N0, N1).astype(np.float32)
    dense0 = tf.keras.layers.Dense(N2)
    tf_y = dense0(np_x)
    np_y = np.matmul(np_x, dense0.kernel.numpy()) + dense0.bias.numpy()
    assert hfe(np_y, tf_y.numpy()) < 1e-5


def test_tf_keras_layers_Dropout():
    dropout_rate = 0.2
    dropout_layer = tf.keras.layers.Dropout(dropout_rate)
    tf0 = tf.random.uniform([1000], 1, 2)
    tf1 = dropout_layer(tf0, training=False) #default False
    assert hfe(tf0.numpy(), tf1.numpy()) < 1e-5
    tf2 = dropout_layer(tf0, training=True)
    assert abs(np.mean(np.abs(tf2.numpy())<0.1) - dropout_rate) < 0.1


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bias = self.add_variable('bias', shape=[num_outputs])

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', shape=[input_shape[-1],self.num_outputs])

    def call(self, x):
        return tf.matmul(x, self.kernel) + self.bias

def test_tf_keras_layers_MyDenseLayer(N0=3, N1=5, N2=7):
    np_x = np.random.randn(N0, N1).astype(np.float32)
    mydense0 = MyDenseLayer(N2)
    tf_y = mydense0(np_x)
    np_y = np.matmul(np_x, mydense0.kernel.numpy()) + mydense0.bias.numpy()
    assert hfe(np_y, tf_y.numpy()) < 1e-5


def test_tf_keras_layers_Embedding(num_embedding=1000, dim_embedding=3, shape=(5,7,11)):
    np0 = np.random.randint(0, num_embedding, size=shape)
    embedding = tf.keras.layers.Embedding(num_embedding, dim_embedding, dtype=tf.float32)
    tf1 = embedding(np0)
    ret_ = embedding.embeddings.numpy()[np0]
    assert hfe(ret_, tf1.numpy()) < 1e-5


def tf_keras_layers_batchnormalization(N0=3, N1=5, momentum=0.7, epsilon=0.1):
    np_gamma0 = np.random.randn(N1).astype(np.float32)
    np_beta0 = np.random.randn(N1).astype(np.float32)
    np_moving_mean0 = np.random.randn(N1).astype(np.float32)
    np_moving_variance0 = np.random.rand(N1).astype(np.float32)
    np_x = np.random.randn(N0, N1).astype(np.float32)
    bn0 = tf.keras.layers.BatchNormalization(axis=1, momentum=momentum, epsilon=epsilon)
    bn0.build((N0,N1))
    bn0.gamma.assign(np_gamma0)
    bn0.beta.assign(np_beta0)
    bn0.moving_mean.assign(np_moving_mean0)
    bn0.moving_variance.assign(np_moving_variance0)

    tf_y = bn0(np_x, training=False)
    np_y = np_gamma0 * (np_x - np_moving_mean0)/np.sqrt(np_moving_variance0+epsilon) + np_beta0
    assert hfe(np_y, tf_y.numpy()) < 1e-5
    print('tf_keras_layers_batchnormalization(training=False): np vs tf: ', hfe_r5(np_y, tf_y.numpy()))

    tf_y = bn0(np_x, training=True)
    np_xmean = np_x.mean(axis=0)
    np_xvariance = np.var(np_x, axis=0)
    np_y = np_gamma0 * (np_x - np_xmean)/np.sqrt(np_xvariance+epsilon) + np_beta0
    np_moving_mean1 = np_moving_mean0*momentum + np_xmean*(1-momentum)
    np_moving_variance1 = np_moving_variance0*momentum + np_xvariance*(1-momentum)
    assert hfe(np_y, tf_y.numpy()) < 1e-5
    assert hfe(np_moving_mean1, bn0.moving_mean.numpy()) < 1e-5
    assert hfe(np_moving_variance1, bn0.moving_variance.numpy()) < 1e-5
