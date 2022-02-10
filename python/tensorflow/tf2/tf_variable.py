import numpy as np
import tensorflow as tf

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)

def tf_variable_typename():
    print('# tf_variable_typename')
    tf1 = tf.Variable(np.random.rand(3,5), dtype=tf.float32)
    tf2 = tf1 + 233
    tf3 = tf.constant(np.random.rand(3,5), dtype=tf.float32)
    print('type(tf.Variable()).__name__: ', type(tf1).__name__) #ResourceVariable
    print('type(tf.Variable()+233).__name__: ', type(tf2).__name__) #EagerTensor
    print('type(tf.constant()).__name__: ', type(tf3).__name__) #EagerTensor

def tf_variable_assign():
    print('# tf_variable_assign')
    np1 = np.array(2.33, dtype=np.float32)
    tf1 = tf.Variable(np1, dtype=tf.float32)
    tf1.assign_add(tf.convert_to_tensor(np1))
    print('assign_add: 2.33+2.33= ', tf1.numpy())
    tf1.assign(tf.convert_to_tensor(np1))
    print('assign: 2.33= ', tf1.numpy())

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.tf1 = tf.Variable(2.33)
        self.tf2 = tf.Variable(4.66, trainable=False)

    def call(self, x):
        return x + self.tf1 + self.tf2

def tf_class_variable():
    print('# tf_class_variable')
    model = MyModel()
    print('model.variables:\n', model.variables)
    print('model.trainable_variables:\n', model.trainable_variables)


if __name__ == "__main__":
    tf_variable_typename()
    print()
    tf_variable_assign()
    print()
    tf_class_variable()
