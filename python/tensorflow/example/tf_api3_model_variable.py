import numpy as np
from tqdm import tqdm
import tensorflow as tf


N0 = 1000
N1 = 10
x_data = np.random.randn(N0, N1)
w = np.random.randn(N1)
b = np.random.randn()
y_data = (x_data*w).sum(axis=1) + b + np.random.uniform(-1,1,(N0,))/10

tmp0 = tf.convert_to_tensor(x_data, tf.float32)
tmp1 = tf.convert_to_tensor(y_data, tf.float32)
ds_train = tf.data.Dataset.from_tensor_slices((tmp0,tmp1)).repeat().shuffle(1000).batch(32)
ds_test = tf.data.Dataset.from_tensor_slices((tmp0,tmp1)).batch(32)


class MyModel(tf.keras.Model):
    def __init__(self, N1:int):
        super().__init__()
        self.w = tf.Variable(np.random.randn(N1)/10, dtype=tf.float32)
        self.b = tf.Variable(np.random.randn()/10, dtype=tf.float32)

    @tf.function
    def call(self, x):
        x = tf.math.reduce_sum(x*self.w, axis=1) + self.b
        return x

model = MyModel(N1)
optimizer = tf.keras.optimizers.Adam()
hf_loss = lambda label,prediction: tf.math.reduce_mean((label-prediction)**2)
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
