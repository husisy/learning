import numpy as np
import tensorflow as tf

np_float = np.float64
tf_float = tf.float64

learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7

N0 = 11
N1 = 3
xdata = np.random.randn(N0,N1)
kernel_ = np.random.randn(N1)
bias_ = np.random.randn()
ydata = np.sum(xdata*kernel_, axis=1) + bias_ + np.random.randn(N0)/100
xdata = xdata.astype(np_float)
ydata = ydata.astype(np_float)

np_kernel = np.random.randn(N1, 1).astype(np_float)
np_bias = np.random.rand(1).astype(np_float)
tf_x = tf.constant(xdata, dtype=tf_float)
tf_y = tf.constant(ydata, dtype=tf_float)

dense0 = tf.keras.layers.Dense(1, dtype=tf_float)
_ = dense0(tf_x) #just to initialize
dense0.kernel.assign(tf.constant(np_kernel, dtype=tf_float))
dense0.bias.assign(tf.constant(np_bias, dtype=tf_float))
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
with tf.GradientTape() as tape:
    loss = tf.math.reduce_mean((dense0(tf_x) - tf_y)**2)
tf_gradient = tape.gradient(loss, dense0.trainable_variables)
tf_optimizer.apply_gradients(zip(tf_gradient, dense0.trainable_variables))
