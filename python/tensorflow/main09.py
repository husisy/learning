'''https://www.tensorflow.org/alpha/guide/eager'''
import numpy as np
import tensorflow as tf

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y,),5)

def np_tf_type_conversion(N0=3):
    np1 = np.random.rand(N0,N0)
    tf1 = tf.constant(np1)
    np2 = np1 @ np1
    tf2 = tf1 @ tf1
    np3 = np.matmul(tf1, tf1)
    tf3 = tf.matmul(np1, np1)
    print('np_tf_type_conversion:: np@ vs tf@: ', hfe_r5(np2,tf2.numpy()))
    print('np_tf_type_conversion:: np.matmul vs tf.matmul: ', hfe_r5(np3,tf3.numpy()))

x0 = tf.Variable([[1]], dtype=tf.float32)
with tf.GradientTape() as tape:
    y0 = x0**2
grad = tape.gradient(y0, x0)
