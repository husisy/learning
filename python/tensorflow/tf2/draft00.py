'''https://www.tensorflow.org/beta/guide/checkpoints'''
import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

logdir = next_tbd_dir()


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train(np,uint8,(60000,28,28))
# y_train(np,uint8,(60000,))
# x_test(np,uint8,(10000,28,28))
# y_test(np,uint8,(10000,))
x_train = x_train/255
x_test = x_test/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense0 = tf.keras.layers.Dense(5, activation='relu')
        self.dense1 = tf.keras.layers.Dense(1)

    def call(self, x):
        '''
        x(tf,float32,(N0,))
        (ret)(tf.float32,(N0,))
        '''
        x = self.dense0(x[:,tf.newaxis])
        x = self.dense1(x)[:,0]
        return x

model = MyModel()

tmp1 = {
    'x': tf.random.normal([10]),
    'y': tf.random.normal([10]),
}
ds_train = tf.data.Dataset.from_tensor_slices(tmp1).repeat(10).batch(2)

optimizer = tf.keras.optimizers.Adam(0.1)
model = MyModel()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, logdir, max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)
'''
mangager(tensorflow.python.training.checkpoint_management.CheckpointManager)
    .latest_checkpoint
        (str): path
        (NoneType)
        tf.train.latest_checkpoint()
    .checkpoints(list,str)
    .save()
        (ret)(str): path
'''


for example in ds_train:
    with tf.GradientTape() as tape:
        predict = model(example['x'])
        loss = tf.reduce_mean(tf.abs(predict - example['y']))
    tmp1 = zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables)
    optimizer.apply_gradients(tmp1)
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 10 == 0:
        save_path = manager.save()

z0 = tf.Variable(tf.constant([2.33]))
tmp0 = tf.train.Checkpoint(dense1=tf.train.Checkpoint(bias=z0))
new_root = tf.train.Checkpoint(model=tmp0)
status = new_root.restore(tf.train.latest_checkpoint(logdir))

z0 = tf.train.load_checkpoint(tf.train.latest_checkpoint(logdir))
z1 = {x:z0.get_tensor(x) for x in z0.get_variable_to_dtype_map().keys()}
