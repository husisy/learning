import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

def demo_tensorboard_keras_sequential():
    (x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)/255
    x_test = x_test.astype(np.float32)/255
    y_train = y_train.astype(np.int64)
    y_test = y_test.astype(np.int64)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    logdir = next_tbd_dir()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
    model.fit(x=x_train, y=y_train, epochs=5, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
    print('run "tensorboard --logdir {}" to view the result'.format(logdir))


def demo_tensorboard_scalar():
    logdir = next_tbd_dir()
    np1 = np.sin(np.linspace(0, 10*np.pi, 1000))
    summary_writer = tf.summary.create_file_writer(logdir)
    with summary_writer.as_default():
        for ind1, x in enumerate(np1):
            tf.summary.scalar('sin-value', x, step=ind1)
    print('run "tensorboard --logdir {}" to view the result'.format(logdir))


def demo_tensorboard_image():
    (x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
    id_to_label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    logdir = next_tbd_dir()
    summary_writer = tf.summary.create_file_writer(logdir)
    with summary_writer.as_default():
        ind1 = np.random.randint(0, x_train.shape[0])
        tf.summary.image(id_to_label[y_train[ind1]], x_train[ind1].reshape((1,28,28,1)), step=0)

        ind1 = np.random.randint(0, len(id_to_label))
        ind2 = np.where(y_train==ind1)[0][:25] #TODO random select
        tf.summary.image(id_to_label[ind1], x_train[ind2].reshape((-1,28,28,1)), max_outputs=25, step=1)

        ind1 = np.random.randint(0, len(id_to_label))
        ind2 = np.where(y_train==ind1)[0][:25] #TODO random select
        tmp1 = x_train[ind2].reshape((5,5,28,28,1)).transpose(0,2,1,3,4).reshape((1,140,140,1))
        tf.summary.image(id_to_label[ind1], tmp1, step=2)
    print('run "tensorboard --logdir {}" to view the result'.format(logdir))

# TODO https://www.tensorflow.org/tensorboard/r2/image_summaries


class MyModel00(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv0 = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.dense0 = tf.keras.layers.Dense(32, activation='relu')
        self.dense1 = tf.keras.layers.Dense(10)

    @tf.function
    def call(self, x):
        x = self.conv0(x)
        tmp1 = x.shape.as_list()[1:]
        x = tf.reshape(x, (-1,tmp1[0]*tmp1[1]*tmp1[2]))
        x = self.dense0(x)
        x = self.dense1(x)
        return x

def demo_tensorboard_tf_keras_model_graph():
    model = MyModel00()
    tf0 = tf.constant(np.random.randn(128, 28, 28, 3).astype(np.float32))
    hf_loss = lambda x: tf.math.reduce_sum(x**2)

    # must initialize in it, hf_loss is not traced
    logdir = next_tbd_dir()
    summary_writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True, profiler=True)
    loss_i = tf.math.reduce_sum(model(tf0)**2)
    with summary_writer.as_default():
        tf.summary.trace_export(name="MyModel00", step=0, profiler_outdir=logdir)
    print('run "tensorboard --logdir {}" to view the result'.format(logdir))
