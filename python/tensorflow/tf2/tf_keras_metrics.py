import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_keras_metrics_mean():
    mean_metrics = tf.keras.metrics.Mean()
    mean_metrics(0)
    mean_metrics(1)
    print('tf_keras_metrics_mean([0,1]): ', mean_metrics.result().numpy())
    mean_metrics([2,3])
    print('tf_keras_metrics_mean([0,1,2,3]): ', mean_metrics.result().numpy())
    mean_metrics.reset_states()
    mean_metrics([4,5])
    print('tf_keras_metrics_mean([4,5]): ', mean_metrics.result().numpy())


if __name__ == "__main__":
    tf_keras_metrics_mean()
