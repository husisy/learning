import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import SimpleRNN, Lambda, LSTM
from keras.losses import mse
from keras.optimizers import SGD
import keras.backend as K

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def keras_SimpleRNN(N0=10, N1=5, N2=7, N3=32):
    np1 = np.random.normal(size=[N0,N1,N2])

    K.clear_session()
    DNN = Sequential([
        SimpleRNN(N3, return_sequences=True, input_shape=(N1,N2)),
        Lambda(lambda x: tf.reduce_mean(x, axis=[1,2])[:,tf.newaxis])
    ])
    DNN.compile(SGD(), loss=mse)
    sess = K.get_session()
    tfG = sess.graph
    tf1 = tfG.get_tensor_by_name('simple_rnn_1_input:0')
    tf2 = tfG.get_tensor_by_name('simple_rnn_1/transpose_1:0')
    ret1 = sess.run(tf2, feed_dict={tf1:np1})

    tmp1 = DNN.layers[0]
    z1 = {x.name.split('/')[1].split(':')[0]:y for x,y in zip(tmp1.weights, tmp1.get_weights())}
    tmp1 = np.concatenate([z1['kernel'],z1['recurrent_kernel']], axis=0)
    ret2 = _tf_rnn_sequence(np1, tmp1, z1['bias'], np.zeros([N0,N3]))
    print('keras_SimpleRNN:: keras vs tf: ', hfe_r5(ret1, ret2))

def _tf_rnn_sequence(np1, np_kernel, np_bias, np_h):
    with tf.Graph().as_default() as tfG:
        tf1 = tf.constant(np1)
        tf_h = tf.constant(np_h)
        rnn0 = tf.nn.rnn_cell.BasicRNNCell(np_kernel.shape[1], name='rnn0')
        tf2, _ = tf.nn.dynamic_rnn(rnn0, tf1, initial_state=tf_h, scope='d_rnn0')
        z1 = {x.name:x for x in rnn0.weights}
        aop = [tf.assign(z1['d_rnn0/rnn0/kernel:0'], np_kernel), tf.assign(z1['d_rnn0/rnn0/bias:0'], np_bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        return sess.run(tf2)


def keras_LSTM(N0=10, N1=5, N2=7, N3=13):
    inputx = np.random.normal(size=[N0,N1,N2])

    K.clear_session()
    DNN = Sequential([
        LSTM(N3, return_sequences=True, recurrent_activation='sigmoid', input_shape=(N1,N2)),
        Lambda(lambda x: tf.reduce_mean(x, axis=[1,2])[:,tf.newaxis])
    ])
    DNN.compile(SGD(), loss=mse)
    sess = K.get_session()
    tfG = sess.graph
    tf1 = tfG.get_tensor_by_name('lstm_1_input:0')
    tf2 = tfG.get_tensor_by_name('lstm_1/transpose_1:0')
    ret1 = sess.run(tf2, feed_dict={tf1:inputx})

    tmp1 = DNN.layers[0]
    def hf1(x):
        x1,x2,x3,x4 = np.split(x, 4, axis=-1)
        return np.concatenate([x1,x3,x2,x4], axis=-1)
    z1 = {x.name.split('/')[1].split(':')[0]:hf1(y) for x,y in zip(tmp1.weights, tmp1.get_weights())}
    tmp1 = np.concatenate([z1['kernel'],z1['recurrent_kernel']], axis=0)
    tmp2 = z1['bias'].copy()
    tmp2[(2*N3):(3*N3)] -= 1 #forget_bias was added here, after mius one here it should be zero
    ret2 = _tf_lstm_sequence(inputx, tmp1, tmp2, np.zeros([N0,N3]), np.zeros([N0,N3]), 1)
    print('keras_LSTM:: keras vs tf: ', hfe_r5(ret1, ret2))

def _tf_lstm_sequence(inputx, kernel, bias, cell, hidden, forget_bias):
    with tf.Graph().as_default() as tfG:
        tf_x = tf.constant(inputx)
        tf_c = tf.constant(cell)
        tf_h = tf.constant(hidden)
        lstm0 = tf.nn.rnn_cell.BasicLSTMCell(hidden.shape[-1], forget_bias, name='lstm0')
        tmp1 = tf.nn.rnn_cell.LSTMStateTuple(tf_c, tf_h)
        tf1,_ = tf.nn.dynamic_rnn(lstm0, tf_x, initial_state=tmp1, scope='d_lstm0')
        z1 = {x.name:x for x in lstm0.weights}
        aop = [tf.assign(z1['d_lstm0/lstm0/kernel:0'], kernel), tf.assign(z1['d_lstm0/lstm0/bias:0'], bias)]
    with tf.Session(graph=tfG) as sess:
        _ = sess.run(aop)
        return sess.run(tf1)


if __name__=='__main__':
    keras_SimpleRNN()
    print()
    keras_LSTM()
