import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

assert tf.executing_eagerly()


def tf_gradient_basic():
    np1 = np.random.rand(3, 4).astype(np.float32)
    tf1 = tf.Variable(np1, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tf2 = tf.reduce_sum(tf1**2/2)
    tf3 = tape.gradient(tf2, tf1)
    print('tf_gradient_basic: ', hfe_r5(np1, tf3.numpy()))


if __name__ == "__main__":
    tf_gradient_basic()
