'''
link:
   * [TensorFlow Datasets](https://github.com/tensorflow/datasets)
   * [tfds-colab](https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb)
'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

_ = tfds.list_builders()

mnist_train = tfds.load(name="mnist", split=tfds.Split.TRAIN)
assert isinstance(mnist_train, tf.data.Dataset)

tmp1,_ = mnist_train.take(count=2)
x1,y1 = tmp1['image'], tmp1['label']
plt.imshow(x1.numpy()[:,:,0])


# another way
mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()
mnist_train = mnist_builder.as_dataset(split=tfds.Split.TRAIN)

info = mnist_builder.info
print(info)
print(info.features)
print(info.features['label'].num_classes)
print(info.features['label'].names)

ds_train = mnist_train.repeat().shuffle(1024).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
