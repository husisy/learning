import numpy as np
import tensorflow as tf

dim_x1x2 = 233 #num_word
shape_x3 = 3 #num_tag
shape_y1 = 5 #num_department

num_data = 1024
x1_data = np.random.randint(dim_x1x2, size=(num_data,10))
x2_data = np.random.randint(dim_x1x2, size=(num_data,100))
x3_data = np.random.randint(2, size=(num_data,shape_x3))

class MyModel(tf.keras.Model):

    def __init__(self, dim_x1x2, shape_x3, shape_y1):
        super().__init__()
        self.dim_x1x2 = dim_x1x2
        self.shape_x3 = shape_x3
        self.shape_y1 = shape_y1
        self.embedding1 = tf.keras.layers.Embedding(dim_x1x2, 64)
        self.lstm1 = tf.keras.layers.LSTM(128)
        self.lstm2 = tf.keras.layers.LSTM(32)

    def call(self, x1, x2, x3):
        x1 = self.embedding1(x1)
        x2 = self.embedding1(x2)

