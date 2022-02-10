import numpy as np
import tensorflow as tf

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_tf_keras_metrics_mean():
    mean_metrics = tf.keras.metrics.Mean()
    mean_metrics(0)
    mean_metrics(1)
    assert hfe(mean_metrics.result().numpy(), 0.5) < 1e-5
    mean_metrics([2,3])
    assert hfe(mean_metrics.result().numpy(), 1.5) < 1e-5
    mean_metrics.reset_states()
    mean_metrics([4,5])
    assert hfe(mean_metrics.result().numpy(), 4.5) < 1e-5
