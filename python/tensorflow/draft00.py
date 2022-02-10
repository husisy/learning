import numpy as np
import tensorflow as tf

# dynamic memory allocation
for x in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(x, True)


# TODO https://www.tensorflow.org/guide/advanced_autodiff

# see where your variables get placed
# tf.debugging.set_log_device_placement(True) #must be set at program start

# misc
tf.__version__
tf.executing_eagerly()
tf.config.list_physical_devices('GPU') #case-sensitive


# tensor
tf0 = tf.constant([2, 23, 233]) #(tensorflow.python.framework.ops.EagerTensor,tf.int32)
tf0.shape #tf0.shape.as_list()
tf0.device
tf0.dtype
tf0.ndim
tf.size(tf0)
tf0.numpy() #(np,int32) #not share memory
tf.constant([True, False, False]) #tf.bool
tf.constant([0.2, 0.23, 0.233]) #tf.float32
tf.constant([0.2j, 0.23j, 0.233j]) #tf.complex128
tf.convert_to_tensor(np.array([0.2, 0.23, 0.233])) #numpy -> tensor
tf.reshape(tf0, [3,1])


# Variable
tf0 = tf.Variable([0.2, 0.23, 0.233]) #(tensorflow.python.ops.resource_variable_ops.ResourceVariable, tf.int32)
tf0.dtype
tf0.shape
tf0.numpy() #(np,int32) #not share memory
tf.convert_to_tensor(tf0) #variable -> tensor
tf0.assign([0.233, 0.2, 0.23]) #in-place operation
tf0.assign_add([2, 23, 233])
tf.Variable(tf0) #new Variable, not share memory
tf0 = tf.Variable([0.2, 0.23, 0.233], trainable=False)
with tf.device('CPU:0'):
    tf1 = tf.Variable([0.2, 0.23, 0.233])


# basic operation
tf0 = tf.random.uniform((2,3), 0, 1)
tf1 = tf.random.uniform((2,3), 0, 1)
tf2 = tf.random.uniform((3,4), 0, 1)
tf.add(tf0, tf1) #tf0+tf1
tf.subtract(tf0, tf1) #tf0-tf1
tf.multiply(tf0, tf1) #tf0*tf1
tf.matmul(tf0, tf2) #tf0@tf2
tf.math.reduce_sum(tf0, axis=1)
tf.math.argmax(tf0, axis=1)


# indexing, support basic indexing
# tensorflow data layout: right-most indexing first, the same as the numpy default
tf.newaxis
tf.reshape #create a view, NOT duplicate the underlying data
tf.transpose #TODO whether it duplicate data


# cast dtype
tf.cast


# broadcasting
tf.broadcast_to #do take up that much of memory


# RaggedTensor
tf0 = tf.ragged.constant([[2], [2,3], [2,3,3]])
tf0.shape


# StringTensor
tf.constant('hello world')
tf.constant(['hello', 'world'])
tf.strings.split
tf.strings.to_number
tf.strings.bytes_split
tf.strings.unicode_split
# TODO unicode tutorial: https://www.tensorflow.org/tutorials/load_data/unicode


# SparseTensor
tf0 = tf.sparse.SparseTensor(indices=[[0,0], [1,2]], values=[1,2], dense_shape=[3,4])
tf.sparse.to_dense(tf0)


# autograd
tf0 = tf.random.normal((2,3), dtype=tf.float32)
tf1 = tf.Variable(np.random.randn(3, 2), dtype=tf.float32)
with tf.GradientTape() as tape:
    # tf2 = tf0 @ tf1 @ tf0 @ tf1 #equivilantly
    tf2 = tf.math.reduce_sum(tf0 @ tf1 @ tf0 @ tf1)
grad = tape.gradient(tf2, tf1)
tape.watched_variables()

tf0 = tf.random.uniform(shape=[])
with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(tf0)
    tf1 = tf0**2
grad = tape.gradient(tf1, tf0)

# TODO persistent=True #performances issue
# TODO with tape.stop_recording()
# TODO tape.reset()
# TODO tf.stop_gradient()


# custom gradient operator
@tf.custom_gradient
def MyOperator00(x):
    y = x**2
    def hf0(dy):
        return -2*dy*x #ni da wo ya
    return y, hf0
tf0 = tf.Variable(0.233, dtype=tf.float32)
with tf.GradientTape() as tape:
    tf1 = MyOperator00(tf0)
grad = tape.gradient(tf1, tf0)
