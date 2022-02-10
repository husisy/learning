import numpy as np
import tensorflow as tf

class ResnetIdentityBlock(tf.keras.Model):
    '''see https://www.tensorflow.org/alpha/tutorials/eager/custom_layers#models_composing_layers'''
    def __init__(self, kernel_size, num_filter1, num_filter2, num_filter3):
        super().__init__(name='')

        self.conv2a = tf.keras.layers.Conv2D(num_filter1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(num_filter2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(num_filter3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        return tf.nn.relu(x+input_tensor)
