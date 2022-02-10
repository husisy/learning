import numpy as np
import tensorflow as tf
from tqdm import tqdm

hf_randn_c = lambda *size: np.random.randn(*size) + np.random.randn(*size) * 1j

N0 = 1000
N1 = 10


x_data = hf_randn_c(N0, N1)
w = hf_randn_c(N1)
b = hf_randn_c()
y_data = (x_data*w).sum(axis=1) + b + hf_randn_c(N0)/10

tmp0 = tf.convert_to_tensor(x_data, dtype=tf.complex64)
tmp1 = tf.convert_to_tensor(y_data, dtype=tf.complex64)
ds_train = tf.data.Dataset.from_tensor_slices((tmp0,tmp1)).repeat().shuffle(10000).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices((tmp0,tmp1)).batch(32)


class MyModel(tf.keras.Model):
    def __init__(self, N1):
        super().__init__()
        self.w_real = tf.Variable(np.random.randn(N1)/10, dtype=tf.float32)
        self.w_imag = tf.Variable(np.random.randn(N1)/10, dtype=tf.float32)
        self.b_real = tf.Variable(np.random.randn()/10, dtype=tf.float32)
        self.b_imag = tf.Variable(np.random.randn()/10, dtype=tf.float32)

    @tf.function
    def call(self, x):
        tmp0 = tf.dtypes.complex(self.w_real, self.w_imag)
        tmp1 = tf.dtypes.complex(self.b_real, self.b_imag)
        ret = tf.math.reduce_sum(x*tmp0, axis=1) + tmp1
        return ret


def hf_loss(prediction, label):
    tmp0 = prediction - label
    ret = tf.math.reduce_mean(tf.math.real(tmp0)**2 + tf.math.imag(tmp0)**2)
    return ret

model = MyModel(N1)
optimizer = tf.keras.optimizers.Adam()
history_loss = []
with tqdm(range(5000)) as pbar:
    for ind_step,(data,label) in zip(pbar, ds_train):
        with tf.GradientTape() as tape:
            prediction = model(data)
            loss_i = hf_loss(label, prediction)
        gradient = tape.gradient(loss_i, model.trainable_variables)
        optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        history_loss.append(float(loss_i.numpy()))
        if (ind_step+1)%50==0:
            pbar.set_postfix(loss=sum(history_loss[-10:])/10)
