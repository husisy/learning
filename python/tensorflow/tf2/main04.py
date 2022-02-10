'''https://www.tensorflow.org/alpha/tutorials/keras/basic_classification'''
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# plt.imshow(train_images[0])

# use tensorflow-datasets to get info
id_to_label = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
model.compile('adam',  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images/255, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_images/255, test_labels)
print('Test accuracy:', test_acc)

prediction_proba = model.predict(test_images/255)
prediction = np.argmax(prediction_proba, axis=1)
test_acc = np.mean(test_labels==prediction)
