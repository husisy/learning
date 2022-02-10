import numpy as np
import tensorflow as tf

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.fc0 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout0 = tf.keras.layers.Dropout(0.2) #keep_probability=1-0.2
        self.fc1 = tf.keras.layers.Dense(10, activation=None)
    @tf.function
    def call(self, x, training=False):
        tmp1 = x.shape.as_list()[1:]
        x = tf.reshape(x, [-1,tmp1[0]*tmp1[1]])
        x = self.fc0(x)
        x = self.dropout0(x, training=training)
        x = self.fc1(x)
        return x

def np_cross_entropy(logits, label):
    tmp0 = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    predict_proba = tmp0/tmp0.sum(axis=-1, keepdims=True)
    loss = -np.mean(np.log(predict_proba[np.arange(len(predict_proba)), label]))
    return loss

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
# x_train(np,uint8,(60000,28,28))
# y_train(np,uint8,(60000,))
# x_test(np,uint8,(10000,28,28))
# y_test(np,uint8,(10000,))
x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28,28)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10),
# ])

model = MyModel()
# model.build((None,28,28))
# model.summary()
# tf.keras.utils.plot_model(model, to_file='model.png') #need pydot and graphviz
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=3, validation_split=0.2)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'loss={loss}, accuracy={accuracy}')

## model by hands
params = {
    'fc0/kernel': model.fc0.variables[0].numpy(),
    'fc0/bias': model.fc0.variables[1].numpy(),
    'fc1/kernel': model.fc1.variables[0].numpy(),
    'fc1/bias': model.fc1.variables[1].numpy(),
}
x = np.matmul(x_test.reshape((x_test.shape[0],-1)), params['fc0/kernel']) + params['fc0/bias']
x = np.maximum(x, 0)
logits = np.matmul(x, params['fc1/kernel']) + params['fc1/bias']
loss = np_cross_entropy(logits, y_test)
acc = np.mean(np.argmax(logits, axis=1)==y_test)
print('loss:{}, accuracy:{}'.format(loss, acc))
