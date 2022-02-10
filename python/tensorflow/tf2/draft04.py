import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

# @tf.function(input_signature(tf.TensorSpec(shape=[None])))
# def test_func(x):
#     pass
