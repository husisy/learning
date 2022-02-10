import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

def tensorboard_scalar():
    logdir = next_tbd_dir()
    np1 = np.sin(np.linspace(0, 10*np.pi, 1000))
    summary_writer = tf.summary.create_file_writer(logdir)
    with summary_writer.as_default():
        for ind1, x in enumerate(np1):
            tf.summary.scalar('sin-value', x, step=ind1)
    print('tensorboard_scalar:: run "tensorboard --logdir {}" to view the result'.format(logdir))


def tensorboard_image():
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
    print('tensorboard_image:: run "tensorboard --logdir {}" to view the result'.format(logdir))

# TODO https://www.tensorflow.org/tensorboard/r2/image_summaries
