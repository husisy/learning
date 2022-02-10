'''https://www.tensorflow.org/tensorboard/r2/hyperparameter_tuning_with_hparams'''
import numpy as np
import tensorflow as tf
from tensorboard.plugins.hparams import api as hparams

from utils import next_tbd_dir


def train_test_model(hparams_param, x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hparams_param['num_unit'], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams_param['dropout_rate']),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])
    model.compile(
        optimizer=hparams_param['optimizer'],
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(x_train, y_train, epochs=2)
    _, accuracy = model.evaluate(x_test, y_test)
    return accuracy


(x_train,y_train), (x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float)/255
x_test = x_test.astype(np.float)/255

NUM_UNIT = hparams.HParam('num_unit', hparams.Discrete([16, 32]))
DROPOUT_RATE = hparams.HParam('dropout_rate', hparams.RealInterval(0.1, 0.2))
OPTIMIZER = hparams.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

logdir = next_tbd_dir()
writer = tf.summary.create_file_writer(logdir)
with writer.as_default():
    hparams.hparams_config(hparams=[NUM_UNIT, DROPOUT_RATE, OPTIMIZER],
            metrics=hparams.Metric('accuracy', display_name='accuracy'))

# TODO
