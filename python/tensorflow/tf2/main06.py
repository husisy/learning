'''https://www.tensorflow.org/alpha/tutorials/keras/feature_columns'''
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
# TODO move tf.feature_column to seperate file

def df_to_dataset(pd1, shuffle=True, batch_size=32):
    pd1 = pd1.copy()
    labels = pd1.pop('target')
    if shuffle:
        return tf.data.Dataset.from_tensor_slices((dict(pd1), labels)) \
                .shuffle(buffer_size=len(pd1)).batch(batch_size)
    else:
        return tf.data.Dataset.from_tensor_slices((dict(pd1), labels)).batch(batch_size)

pd1 = pd.read_csv('https://storage.googleapis.com/applied-dl/heart.csv')
train, test = train_test_split(pd1, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

ds_test = df_to_dataset(train, batch_size=5)
ds_train = df_to_dataset(train, batch_size=32)
ds_val = df_to_dataset(val, shuffle=False, batch_size=32)
ds_test = df_to_dataset(test, shuffle=False, batch_size=32)

for feature_batch, label_batch in ds_test.take(2):
    print('Every feature:', list(feature_batch.keys()))
    print('A batch of ages:', feature_batch['age'])
    print('A batch of targets:', label_batch )

# numeric column
print(next(iter(ds_test))[0]['age'].numpy())
tmp1 = tf.feature_column.numeric_column('age')
tmp2 = tf.keras.layers.DenseFeatures(tmp1)
tmp2(next(iter(ds_test))[0]).numpy()

# bucketized column
print(next(iter(ds_test))[0]['age'].numpy())
tmp1 = tf.feature_column.numeric_column('age')
tmp2 = tf.feature_column.bucketized_column(tmp1, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
tmp3 = tf.keras.layers.DenseFeatures(tmp2)
tmp3(next(iter(ds_test))[0]).numpy()

# categorical column
print(next(iter(ds_test))[0]['thal'].numpy())
tmp1 = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
tmp2 = tf.feature_column.indicator_column(tmp1)
tmp3 = tf.keras.layers.DenseFeatures(tmp2)
tmp3(next(iter(ds_test))[0]).numpy()

# embedding column
print(next(iter(ds_test))[0]['thal'].numpy())
tmp1 = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
tmp2 = tf.feature_column.embedding_column(tmp1, dimension=8)
tmp3 = tf.keras.layers.DenseFeatures(tmp2)
tmp3(next(iter(ds_test))[0]).numpy()

# hashed feature column
print(next(iter(ds_test))[0]['thal'].numpy())
tmp1 = tf.feature_column.categorical_column_with_hash_bucket('thal', hash_bucket_size=1000)
tmp2 = tf.feature_column.indicator_column(tmp1)
tmp3 = tf.keras.layers.DenseFeatures(tmp2)
tmp3(next(iter(ds_test))[0]).numpy()

# crossed feature column
tmp1 = tf.feature_column.numeric_column('age')
tmp2 = tf.feature_column.bucketized_column(tmp1, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
tmp3 = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
tmp4 = tf.feature_column.crossed_column([tmp2,tmp3], hash_bucket_size=1000)
tmp5 = tf.feature_column.indicator_column(tmp4)
tmp6 = tf.keras.layers.DenseFeatures(tmp5)
tmp6(next(iter(ds_test))[0]).numpy()


tmp1 = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']
feature_columns = [tf.feature_column.numeric_column(x) for x in tmp1]

# yaoezi
tmp1 = tf.feature_column.numeric_column('age')
tmp2 = tf.feature_column.bucketized_column(tmp1, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
tmp3 = tf.feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'])
tmp4 = tf.feature_column.crossed_column([tmp2, tmp3], hash_bucket_size=1000)
feature_columns.append(tmp2)
feature_columns.append(tf.feature_column.indicator_column(tmp3))
feature_columns.append(tf.feature_column.embedding_column(tmp3, dimension=8))
feature_columns.append(tf.feature_column.indicator_column(tmp4))

model = tf.keras.Sequential([
  tf.keras.layers.DenseFeatures(feature_columns),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile('adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(ds_train, validation_data=ds_val, epochs=5)

loss, accuracy = model.evaluate(ds_test)
print("Accuracy", accuracy)
