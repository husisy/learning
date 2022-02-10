'''https://www.tensorflow.org/beta/guide/checkpoints'''
import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

logdir = next_tbd_dir()
hf_file = lambda *x,_dir=logdir: os.path.join(_dir, *x)


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

zc0 = tf.Variable(tf.constant([2.33]))
tmp0 = tf.train.Checkpoint(dense1=tf.train.Checkpoint(bias=zc0))
new_root = tf.train.Checkpoint(model=tmp0)
status = new_root.restore(tf.train.latest_checkpoint(logdir))

zc0 = tf.train.load_checkpoint(tf.train.latest_checkpoint(logdir))
zc1 = {x:zc0.get_tensor(x) for x in zc0.get_variable_to_dtype_map().keys()}
