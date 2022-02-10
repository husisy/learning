from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

## mnist
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()


## iris
tmp0 = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv'
filepath = tf.keras.utils.get_file(fname='iris_training.csv', origin=tmp0)
pd0 = pd.read_csv(
    filepath,
    skiprows=1,
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
)
id_to_label = {0:'Iris setosa', 1:'Iris versicolor', 2:'Iris virginica'}
ds_train_iris = tf.data.Dataset.from_tensor_slices((pd0.iloc[:,:4].values, pd0.iloc[:,4].values)) #TODO pandas indexing

tmp0 = 'https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv'
filepath = tf.keras.utils.get_file(fname='iris_test.csv', origin=tmp0)
pd0 = pd.read_csv(
    filepath,
    skiprows=1,
    names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'],
)
ds_test_iris = tf.data.Dataset.from_tensor_slices((pd0.iloc[:,:4].values, pd0.iloc[:,4].values))


## flower_photos
tmp0 = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
filepath = Path(tf.keras.utils.get_file(origin=tmp0, fname='flower_photos', untar=True))
data = [(x.parent.stem, x) for x in filepath.glob('*/*')]
