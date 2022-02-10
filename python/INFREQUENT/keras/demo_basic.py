import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Dense, Activation, Lambda
from keras.optimizers import SGD
from keras.models import Sequential
from keras.losses import categorical_crossentropy, mse

from utils import next_tbd_dir

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)


def keras_eval_predict(N0=1000, N1=100, N2=10):
    X_train = np.random.normal(size=[N0,N1])
    tmp1 = np.random.randint(0, N2, [N0])
    y_train = np.zeros([N0,N2], dtype=np.int64)
    y_train[np.arange(N0), tmp1] = 1

    DNN = Sequential([
        Dense(64, input_dim=N1),
        Activation('relu'),
        Dense(N2),
        Activation('softmax'),
    ])
    DNN.compile(loss=categorical_crossentropy, optimizer=SGD(), metrics=['accuracy'])
    # DNN.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    DNN.fit(X_train, y_train, epochs=2, batch_size=32)
    eval_loss,eval_acc = DNN.evaluate(X_train, y_train, batch_size=128)

    predict_proba = DNN.predict(X_train, batch_size=128)
    tmp1 = np.argmax(y_train, axis=1)
    pred_loss = -np.mean(np.log(predict_proba[np.arange(N0), tmp1]))
    pred_acc = np.mean(np.argmax(predict_proba, axis=1)==tmp1)
    print('keras_eval_predict loss:: np vs keras: ', hfe_r5(eval_loss, pred_loss))
    print('keras_eval_predict acc :: np vs keras: ', hfe_r5(eval_acc, pred_acc))


def keras_session_graph(dir0):
    sess = K.get_session()
    tfG = sess.graph
    tf.summary.FileWriter(dir0, tfG).close()

def _test_keras_session_graph():
    K.clear_session()
    DNN = Sequential()
    DNN.add(Dense(64, activation='relu', input_dim=100))
    DNN.add(Dense(10, activation='softmax'))
    DNN.compile(loss=categorical_crossentropy, optimizer=SGD(), metrics=['accuracy'])
    logdir = next_tbd_dir()
    keras_session_graph(logdir)
    print('please run "tensorboard --logdir {}" to sess graph'.format(logdir))


def keras_Lambda_Print(N0=10):
    X_train = np.random.normal(size=[N0,1])
    y_train = np.random.normal(size=[N0])
    DNN = Sequential([
        Dense(1, input_shape=[1]),
        Lambda(lambda x: tf.Print(x, ['ni da wo ya: ', -x])),
    ])
    DNN.compile(SGD(), loss=mse, metrics=['mae'])
    DNN.fit(X_train, y_train, epochs=2, batch_size=3)


if __name__=='__main__':
    keras_eval_predict()
    print()
    _test_keras_session_graph()
    print()
    keras_Lambda_Print()
