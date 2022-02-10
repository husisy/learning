'''https://www.tensorflow.org/tutorials/quickstart/advanced'''
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv0 = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.dense0 = tf.keras.layers.Dense(32, activation='relu')
        self.dense1 = tf.keras.layers.Dense(10)

    @tf.function
    def call(self, x):
        x = self.conv0(x)
        tmp1 = x.shape.as_list()[1:]
        x = tf.reshape(x, (-1,tmp1[0]*tmp1[1]*tmp1[2]))
        x = self.dense0(x)
        x = self.dense1(x)
        return x

# TODO see pytorch example
# TODO len_trainset idoit, len(tf.data.Dataset) will be added in tf-2.3.0
# TODO lr_scheduler
# TODO training=True/False

batchsize = 128 #for progressbar
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train[:,:,:,np.newaxis].astype(np.float32)/255 #(np,float32,(60000,28,28,1))
x_test = x_test[:,:,:,np.newaxis].astype(np.float32)/255 #(np,float32,(60000,28,28,1))
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)
trainloader = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(len(x_train)).batch(batchsize)
testloader = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(batchsize)
len_trainloader = int(np.ceil(x_train.shape[0]/batchsize).item())
len_testloader = int(np.ceil(x_test.shape[0]/batchsize).item())

model = MyModel()
hf_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

metric_history = defaultdict(list)
for ind_epoch in range(3):
    with tqdm(total=len_trainloader, desc='epoch-{}'.format(ind_epoch)) as pbar:
        train_correct = 0
        train_total = 0
        for ind_batch,(data_i,label_i) in enumerate(trainloader):
            with tf.GradientTape() as tape:
                prediction = model(data_i)
                loss_i = hf_loss(label_i, prediction)
            gradient = tape.gradient(loss_i, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))

            train_correct += tf.math.reduce_sum(tf.cast(tf.argmax(prediction, axis=1)==label_i, tf.int64)).numpy().item()
            train_total += label_i.shape[0]
            metric_history['train-loss'].append(loss_i.numpy().item())
            if ind_batch+1 < len_trainloader:
                pbar.set_postfix({
                    'loss':'{:5.3}'.format(loss_i.numpy().item()),
                    'acc':'{:4.3}%'.format(100*train_correct/train_total),
                })
                pbar.update() #move the last update to val
        metric_history['train-acc'].append(train_correct / train_total)

        test_correct = 0
        test_total = 0
        for data_i,label_i in testloader:
            prediction = model(data_i)
            test_correct += tf.math.reduce_sum(tf.cast(tf.argmax(prediction, axis=1)==label_i, tf.int64)).numpy().item()
            test_total += label_i.shape[0]
        metric_history['test-acc'].append(test_correct / test_total)
        pbar.set_postfix({
            'acc':'{:4.3}%'.format(100*train_correct/train_total),
            'test-acc':'{:4.3}%'.format(100*test_correct / test_total),
        })
        pbar.update()
