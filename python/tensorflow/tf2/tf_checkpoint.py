import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_checkpoint_basic():
    checkpoint_prefix = next_tbd_dir() + os.sep
    np1 = np.random.rand(3, 5).astype(np.float32)
    tf1 = tf.Variable(np1, dtype=tf.float32)
    checkpoint = tf.train.Checkpoint(tf1=tf1)
    checkpoint.save(checkpoint_prefix)
    tf1.assign(np.random.rand(*tf1.shape.as_list()))
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))
    print('tf_checkpoint_basic: ', hfe_r5(np1, tf1.numpy()))


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # x(tf,float32,(None,32,32))
        tmp1 = x.shape.as_list()[1:]
        x = tf.reshape(x, [-1,tmp1[0]*tmp1[1]])
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def tf_checkpoint_advance():
    checkpoint_prefix = next_tbd_dir() + os.sep
    np1 = np.random.rand(3, 28, 28)

    model0 = MyModel()
    model0.build((None,28,28))
    optimizer0 = tf.keras.optimizers.Adam()
    model0.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer0, metrics=['accuracy'])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer0, model=model0)
    checkpoint.save(checkpoint_prefix)
    tf0 = model0(np1)

    model1 = MyModel()
    model1.build((None,28,28))
    optimizer1 = tf.keras.optimizers.Adam()
    model1.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    checkpoint = tf.train.Checkpoint(optimizer=optimizer1, model=model1)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))
    tf1 = model1(np1)

    print('tf_checkpoint_advance: ', hfe_r5(tf0.numpy(), tf1.numpy()))


if __name__ == "__main__":
    tf_checkpoint_basic()
    print()
    tf_checkpoint_advance()

