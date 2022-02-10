import os
import numpy as np
import tensorflow as tf


hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)
hf_file = lambda *x: os.path.join('..', 'tbd01', *x)

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        # x(tf,float32,(None,32,32))
        tmp1 = x.shape.as_list()[1:]
        x = tf.reshape(x, [-1,tmp1[0]*tmp1[1]])
        x = self.dense1(x)
        x = self.dense2(x)
        return x

model = MyModel()
# model.build((None,28,28))
# model.summary()
# tf.keras.utils.plot_model(model, to_file=hf_file('model.png'))
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, validation_split=0.2)
print('test_loss: {},\t test_accuracy: {}'.format(*model.evaluate(x_test, y_test)))

# model serialization
model.save_weights(hf_file('tbd01.h5'))
new_model = MyModel()
new_model.build((None,28,28))
new_model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
new_model.load_weights(hf_file('tbd01.h5'))
print('[NEW model] test_loss: {},\t test_accuracy: {}'.format(*new_model.evaluate(x_test, y_test)))

# model by hands
tmp1 = model.dense1.get_weights()
tmp2 = model.dense2.get_weights()
params = {
    'dense1/kernel': tmp1[0],
    'dense1/bias': tmp1[1],
    'dense2/kernel': tmp2[0],
    'dense2/bias': tmp2[1],
}
tmp1 = np.matmul(x_test.reshape((x_test.shape[0],-1)), params['dense1/kernel']) + params['dense1/bias']
tmp1 = np.maximum(tmp1, 0)
tmp1 = np.matmul(tmp1, params['dense2/kernel']) + params['dense2/bias']
logits = tmp1 - tmp1.max(axis=1, keepdims=True)
tmp1 = np.exp(logits)
predict_proba = tmp1 / tmp1.sum(axis=1, keepdims=True)
loss = np.mean(-np.log(predict_proba[np.arange(y_test.shape[0]), y_test.astype(np.int32)]))
acc = np.mean(np.argmax(tmp1, axis=1)==y_test)
print('[by HAND] test_loss: {},\t test_accuracy: {}'.format(loss, acc))
