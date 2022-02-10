'''https://www.tensorflow.org/alpha/tutorials/keras/save_and_restore_models'''
import os
import numpy as np
import tensorflow as tf

from utils import next_tbd_dir

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

logdir = next_tbd_dir()
hf_file = lambda *x,_dir=logdir: os.path.join(_dir, *x)

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

num1 = 1000
train_images = train_images.reshape((-1,784))[:num1]/255
test_images = test_images.reshape((-1,784))[:num1]/255
train_labels = train_labels[:num1]
test_labels = test_labels[:num1]

model = create_model()
checkpoint_path = hf_file('cp.ckpt')
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,  save_weights_only=True, verbose=1)
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images,test_labels), callbacks=[cp_callback])
print(model.evaluate(test_images, test_labels))

model = create_model()
print(model.evaluate(test_images, test_labels))

model = create_model()
model.load_weights(checkpoint_path)
print(model.evaluate(test_images, test_labels))
