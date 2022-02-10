'''
[tf2 - Hello World basic](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/quickstart/beginner.ipynb)
'''
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train(np,uint8,(60000,28,28))
# y_train(np,uint8,(60000,))
# x_test(np,uint8,(10000,28,28))
# y_test(np,uint8,(10000,))
# plt.imshow(x_train[0])
x_train = x_train/255
x_test = x_test/255

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)
