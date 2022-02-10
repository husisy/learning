import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

assert tf.executing_eagerly()


def tf_matmul():
    np1 = np.random.rand(3, 4).astype(np.float32)
    np2 = np.random.rand(4, 5).astype(np.float32)
    np3 = np.matmul(np1, np2)
    tf3 = tf.matmul(tf.convert_to_tensor(np1), tf.convert_to_tensor(np2))
    print('tf_matmul: np vs tf: ', hfe_r5(tf3.numpy(), np3))


if __name__ == "__main__":
    tf_matmul()
