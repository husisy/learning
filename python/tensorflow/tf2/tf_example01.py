'''https://github.com/tensorflow/tensorboard/blob/master/docs/r2/get_started.ipynb'''
import numpy as np
from tqdm import tqdm
import tensorflow as tf


@tf.function
def train_step(image, label, model, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        prediction = model(image)
        loss = loss_object(label, prediction)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    train_loss(loss)
    train_accuracy(label, prediction)


@tf.function
def test_step(image, label, model, loss_object, test_loss, test_accuracy):
    prediction = model(image)
    loss = loss_object(label, prediction)

    test_loss(loss)
    test_accuracy(label, prediction)


(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:,:,:,np.newaxis].astype(np.float32)/255 #(np,float32,(60000,28,28,1))
x_test = x_test[:,:,:,np.newaxis].astype(np.float32)/255 #(np,float32,(60000,28,28,1))

trarin_batchsize = 32

ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(trarin_batchsize)
ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        tmp1 = x.shape.as_list()[1:]
        x = tf.reshape(x, (-1,tmp1[0]*tmp1[1]*tmp1[2]))
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MyModel()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

for ind_epoch in range(2):
    for (image,label),_ in zip(ds_train, tqdm(range(x_train.shape[0]//trarin_batchsize - 1))):
        train_step(image, label, model, loss_object, optimizer, train_loss, train_accuracy)

    for image,label in ds_test:
        test_step(image, label, model, loss_object, test_loss, test_accuracy)

    print('\nepoch: {}\nloss: {}\naccuracy: {}\ntest_loss: {}\ntest_accuracy: {}'.format(ind_epoch,
            train_loss.result(), train_accuracy.result()*100, test_loss.result(), test_accuracy.result()*100))
    # Reset metrics every epoch
    train_loss.reset_states()
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
