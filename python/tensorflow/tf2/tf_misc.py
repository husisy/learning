import numpy as np
import tensorflow as tf

print('tf.__version__: ', tf.__version__)
print('tf.executing_eagerly(): ', tf.executing_eagerly())
print('tf.test.is_gpu_available(): ', tf.test.is_gpu_available())

def tf_gpu_test():
    if tf.test.is_gpu_available():
        with tf.device('gpu:0'):
            _ = tf.Variable(tf.random.normal([1000, 1000]))

