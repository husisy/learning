import numpy as np
import tensorflow as tf


class MyModel00(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.tf0 = tf.Variable(2.33)
        self.tf1 = tf.Variable(4.66, trainable=False)
    def call(self, x):
        ret = x + self.tf0 + self.tf1
        return ret
net = MyModel00()
net.variables
net.trainable_variables
