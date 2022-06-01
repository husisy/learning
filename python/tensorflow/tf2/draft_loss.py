import numpy as np
import tensorflow as tf

np_rng = np.random.default_rng()

def test_hinge():
    size = 5
    y_label = np_rng.choice(np.array([-1,1]), size=size).astype(np.float32)
    y_prediction = np_rng.uniform(-2, 2, size=size).astype(np.float32)
    loss = tf.keras.losses.Hinge()
    ret0 = loss(tf.convert_to_tensor(y_label), tf.convert_to_tensor(y_prediction)).numpy()
    ret_ = np.maximum(1-y_label*y_prediction, 0).mean()
    assert abs(ret0-ret_)<1e-5
