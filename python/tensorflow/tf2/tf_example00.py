'''https://www.tensorflow.org/alpha/tutorials/quickstart/beginner'''
import numpy as np
import tensorflow as tf

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
# x_train(np,uint8,(60000,28,28))
# y_train(np,uint8,(60000,))
# x_test(np,uint8,(10000,28,28))
# y_test(np,uint8,(10000,))
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

DNN = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2), #keep_probability=1-0.2
    tf.keras.layers.Dense(10, activation='softmax'),
])

DNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
DNN.fit(x_train, y_train, epochs=5)
tmp1 = DNN.evaluate(x_test, y_test)
print('loss: {}\naccuracy: {}'.format(*tmp1))
print('manual acc:', np.mean(DNN.predict(x_test).argmax(axis=1)==y_test))
