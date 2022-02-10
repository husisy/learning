import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN, Conv2D, BatchNormalization

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y:round(hfe(x,y), 5)


def tf_keras_layers_Dense(N0=3, N1=5, N2=7):
    np_x = np.random.randn(N0, N1).astype(np.float32)
    dense0 = Dense(N2)
    # dense0.variables
    # dense0.trainable_variables
    tf_y = dense0(np_x)
    np_y = np.matmul(np_x, dense0.kernel.numpy()) + dense0.bias.numpy()
    print('tf_keras_layers_dense: np vs tf: ', hfe_r5(np_y, tf_y.numpy()))


class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.bias = self.add_variable('bias', shape=[num_outputs])

    def build(self, input_shape):
        self.kernel = self.add_variable('kernel', shape=[input_shape[-1],self.num_outputs])

    def call(self, x):
        return tf.matmul(x, self.kernel) + self.bias

def tf_keras_layers_MyDenseLayer(N0=3, N1=5, N2=7):
    np_x = np.random.randn(N0, N1).astype(np.float32)
    mydense0 = MyDenseLayer(N2)
    # mydense0.variables
    # mydense0.trainable_variables
    tf_y = mydense0(np_x)
    np_y = np.matmul(np_x, mydense0.kernel.numpy()) + mydense0.bias.numpy()
    print('tf_keras_layers_MyDenseLayer: np vs tf: ', hfe_r5(np_y, tf_y.numpy()))


def tf_keras_layers_Embedding(num_embedding=1000, dim_embedding=3, shape=(5,7,11)):
    np0 = np.random.randint(0, num_embedding, size=shape)
    embedding = Embedding(num_embedding, dim_embedding)
    tf1 = embedding(np0)
    word_vector = embedding.embeddings.numpy()
    np1 = word_vector[np0]
    print('tf_keras_layers_Embedding: tf vs np: ', hfe_r5(np1, tf1.numpy()))


def tf_keras_layers_conv2d(N0=3, image_size=(11,13), image_channel=3, num_filter=7, kernel_size=(3,5), strides=(2,1), padding='valid'):
    import np_conv2d
    conv0 = Conv2D(num_filter, kernel_size, strides, padding)
    np_image = np.random.randn(N0, image_size[0], image_size[1], image_channel).astype(np.float32)
    tf_y = conv0(tf.convert_to_tensor(np_image))
    np_y = np_conv2d.conv2d(np_image, conv0.kernel.numpy(), padding, strides) + conv0.bias.numpy()
    print('tf_keras_layers_conv2d: np vs tf: ', hfe_r5(np_y, tf_y.numpy()))


def tf_keras_layers_batchnormalization(N0=3, N1=5, momentum=0.7, epsilon=0.1):
    np_gamma0 = np.random.randn(N1).astype(np.float32)
    np_beta0 = np.random.randn(N1).astype(np.float32)
    np_moving_mean0 = np.random.randn(N1).astype(np.float32)
    np_moving_variance0 = np.random.rand(N1).astype(np.float32)
    np_x = np.random.randn(N0, N1).astype(np.float32)
    bn0 = BatchNormalization(axis=1, momentum=momentum, epsilon=epsilon)
    bn0.build((N0,N1))
    bn0.gamma.assign(np_gamma0)
    bn0.beta.assign(np_beta0)
    bn0.moving_mean.assign(np_moving_mean0)
    bn0.moving_variance.assign(np_moving_variance0)

    tf_y = bn0(np_x, training=False)
    np_y = np_gamma0 * (np_x - np_moving_mean0)/np.sqrt(np_moving_variance0+epsilon) + np_beta0
    print('tf_keras_layers_batchnormalization(training=False): np vs tf: ', hfe_r5(np_y, tf_y.numpy()))

    tf_y = bn0(np_x, training=True)
    np_xmean = np_x.mean(axis=0)
    np_xvariance = np.var(np_x, axis=0)
    np_y = np_gamma0 * (np_x - np_xmean)/np.sqrt(np_xvariance+epsilon) + np_beta0
    np_moving_mean1 = np_moving_mean0*momentum + np_xmean*(1-momentum)
    np_moving_variance1 = np_moving_variance0*momentum + np_xvariance*(1-momentum)
    print('tf_keras_layers_batchnormalization(training=True): np vs tf: ', hfe_r5(np_y, tf_y.numpy()))
    print('tf_keras_layers_batchnormalization(training=True) moving_mean: np vs tf: ', hfe_r5(np_moving_mean1, bn0.moving_mean.numpy()))
    print('tf_keras_layers_batchnormalization(training=True) moving_variance: np vs tf: ', hfe_r5(np_moving_variance1, bn0.moving_variance.numpy()))

# N0 = 3
# N1 = 5
# N2 = 7
# N3 = 11
# x_input = np.random.rand(N0, N1, N2).astype(np.float32)
# sequence_length = np.random.randint(2, N1, size=[N0])
# mask = tf.sequence_mask(sequence_length, N1)
# initial_state = np.random.randn(N0, N3).astype(np.float32)

# rnn0 = SimpleRNN(N3, return_sequences=True)
# # tf.convert_to_tensor(initial_state)
# tf1 = rnn0(x_input, initial_state=tf.convert_to_tensor(initial_state), mask=mask)

if __name__ == "__main__":
    tf_keras_layers_Dense()
    print()
    tf_keras_layers_MyDenseLayer()
    print()
    tf_keras_layers_Embedding()
    print()
    tf_keras_layers_conv2d()
    print()
    tf_keras_layers_batchnormalization()
