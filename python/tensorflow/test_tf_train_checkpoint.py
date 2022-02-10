import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_tf_train_checkpoint_basic():
    checkpoint_prefix = next_tbd_dir() + os.sep
    np1 = np.random.rand(3, 5).astype(np.float32)
    tf1 = tf.Variable(np1, dtype=tf.float32)
    checkpoint = tf.train.Checkpoint(tf1=tf1)
    checkpoint.save(checkpoint_prefix)
    tf1.assign(np.random.rand(*tf1.shape.as_list()))
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))
    assert hfe(np1, tf1.numpy()) < 1e-5


class MyModel00(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # x(tf,float32,(None,233))
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def test_tf_train_checkpoint_advance():
    checkpoint_prefix = next_tbd_dir() + os.sep
    np1 = np.random.rand(3, 233).astype(np.float32)

    model0 = MyModel00()
    model0.build((None,233))
    optimizer0 = tf.keras.optimizers.Adam()
    model0.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer0, metrics=['accuracy'])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer0, model=model0)
    checkpoint.save(checkpoint_prefix)
    tf0 = model0(np1)

    model1 = MyModel00()
    model1.build((None,233))
    optimizer1 = tf.keras.optimizers.Adam()
    model1.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer1, model=model1)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))
    tf1 = model1(np1)

    assert hfe(tf0.numpy(), tf1.numpy()) < 1e-5
