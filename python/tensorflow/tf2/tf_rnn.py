import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

def tf_rnncell(N0=3, N1=5, N2=7):
    rnn_cell = tf.keras.layers.SimpleRNNCell(N2)
    np_x = np.random.rand(N0, N1).astype(np.float32)
    np_state = [np.random.rand(N0,N2).astype(np.float32)]
    tf_x = tf.convert_to_tensor(np_x)
    tf_state = [tf.convert_to_tensor(np_state[0])]

    tf_ret = rnn_cell(tf_x, tf_state)
    np_xkernel,np_statekernel,np_bias = rnn_cell.get_weights()
    np_ret = np.tanh(np.matmul(np_x, np_xkernel) + np.matmul(np_state[0], np_statekernel) + np_bias)

    print('tf_rnncell_output: ', hfe_r5(np_ret, tf_ret[0].numpy()))
    print('tf_rnncell_state: ', hfe_r5(np_ret, tf_ret[1][0].numpy()))

    # export graph
    # @tf.function
    # def rnn_fn(x, state):
    #     return rnn_cell(x, state)
    # logdir = next_tbd_dir()
    # writer = tf.summary.create_file_writer(logdir)
    # tf.summary.trace_on(graph=True)
    # rnn_fn(tf_x, tf_state)
    # with writer.as_default():
    #     tf.summary.trace_export(name='rnn_cell', step=0)

if __name__ == "__main__":
    tf_rnncell()
