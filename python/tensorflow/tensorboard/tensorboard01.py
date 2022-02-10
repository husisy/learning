'''https://github.com/tensorflow/tensorboard/blob/master/docs/r2/get_started.ipynb'''
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout

from utils import next_tbd_dir

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = Dense(512, activation='relu')
        self.dropout = Dropout(0.2)
        self.dense2 = Dense(10, activation='softmax')

    def call(self, x, training):
        # x(tf,float32,(N0,28,28))
        tmp1 = x.shape.as_list()[1:]
        x = tf.reshape(x, (-1,tmp1[0]*tmp1[1]))
        x = self.dense1(x)
        if training:
            x = self.dropout(x)
        x = self.dense2(x)
        return x

(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:,:,:,np.newaxis].astype(np.float32)/255 #(np,float32,(60000,28,28,1))
x_test = x_test[:,:,:,np.newaxis].astype(np.float32)/255 #(np,float32,(60000,28,28,1))

batchsize = 32
ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(60000).batch(batchsize)
ds_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batchsize)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')

@tf.function
def train_step(model, optimizer, x_train, y_train):
    with tf.GradientTape() as tape:
        prediction = model(x_train, True)
        loss = loss_object(y_train, prediction)
    gradient = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_train, prediction)

@tf.function
def test_step(model, x_test, y_test):
    prediction = model(x_test, False)
    loss = loss_object(y_test, prediction)

    test_loss(loss)
    test_accuracy(y_test, prediction)

logdir = next_tbd_dir()
train_logdir = os.path.join(logdir, 'train')
test_logdir = os.path.join(logdir, 'test')
train_summary_writer = tf.summary.create_file_writer(train_logdir)
test_summary_writer = tf.summary.create_file_writer(test_logdir)

for ind_epoch in range(5):
    for x_train,y_train in ds_train:
        train_step(model, optimizer, x_train, y_train)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=ind_epoch)
        tf.summary.scalar('accuracy', train_accuracy.result(), step=ind_epoch)

    for x_test,y_test in ds_test:
        test_step(model, x_test, y_test)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=ind_epoch)
        tf.summary.scalar('accuracy', test_accuracy.result(), step=ind_epoch)

    tmp1 = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(tmp1.format(ind_epoch, train_loss.result(), train_accuracy.result()*100,
            test_loss.result(), test_accuracy.result()*100))

    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
