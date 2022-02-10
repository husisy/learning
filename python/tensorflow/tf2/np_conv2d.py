import numpy as np
import tensorflow as tf
# reference: https://github.com/renmengye/np-conv2d
# TODO change to np.lib.stride_tricks.as_strided for efficiency
# see np_vs_xx/tensorflow/zcprivate/numpyflow/conv2d.py

def conv2d(x, w, padding='same', stride=(1,1), floor_first=True):
    """2D convolution (technically speaking, correlation).

    x (np,float32,(N0,H1,W1,C1))
    w (np,float32,(H2,W2,C1,C2))
    padding (str or integer or list of integer): 'same', 'valid'
    stride (integer or list of integer): (SH,SW)

    (ret)(np,float32,(N0,H3,W3,C2))
    """
    padding = [padding,padding] if isinstance(padding, int) else padding
    stride = [stride,stride] if isinstance(stride, int) else stride

    N0,H1,W1,C1 = x.shape
    H2,W2,C1_,C2 = w.shape
    if C1 != C1_:
        print('xs, ws, in-channel not match')
        raise ValueError()

    if padding=='same':
        H3 = np.ceil(H1 / stride[0]).astype(np.int64)
        PH = (H3 - 1)*stride[0] + H2 - H1
        W3 = np.ceil(W1 / stride[1]).astype(np.int64)
        PW = (W3 - 1)*stride[1] + W2 - W1
    elif padding=='valid':
        H3 = np.ceil((H1 - H2 + 1) / stride[0]).astype(np.int64)
        PH = 0
        W3 = np.ceil((W1 - W2 + 1) / stride[1]).astype(np.int64)
        PW = 0
    else:
        PH,PW = padding
        H3 = int(np.ceil((H1 - H2 + PH + 1) / stride[0]))
        W3 = int(np.ceil((W1 - W2 + PW + 1) / stride[1]))
    tmp1 = PH//2 if floor_first else PH-PH//2
    tmp2 = PW//2 if floor_first else PW-PW//2
    padding = [[0,0],[tmp1,PH-tmp1],[tmp2,PW-tmp2],[0,0]]

    x = np.pad(x, padding, mode='constant', constant_values=0)

    y = np.zeros([N0, H3, W3, H2, W2, C1])
    for ind1 in range(H3):
        for ind2 in range(W3):
            ind3 = slice(ind1*stride[0], ind1*stride[0]+H2)
            ind4 = slice(ind2*stride[1], ind2*stride[1]+W2)
            y[:, ind1, ind2, :, :, :] = x[:, ind3, ind4, :]

    tmp1 = y.reshape([N0*H3*W3, H2*W2*C1])
    tmp2 = w.reshape([H2*W2*C1, C2])
    return np.dot(tmp1,tmp2).reshape([N0, H3, W3, C2])


def conv2d_gradw(x, dy, ksize, padding='same', stride=(1, 1), floor_first=True):
    padding = [padding,padding] if isinstance(padding, int) else padding
    stride = [stride,stride] if isinstance(stride, int) else stride

    N0,H1,W1,C1 = x.shape
    H2,W2 = ksize
    _,H3,W3,C2 = dy.shape

    if padding=='same':
        PH = (H3 - 1)*stride[0] + H2 - H1
        PW = (W3 - 1)*stride[1] + W2 - W1
    elif padding=='valid':
        PH = 0
        PW = 0
    else:
        PH,PW = padding
    tmp1 = PH//2 if floor_first else PH-PH//2
    tmp2 = PW//2 if floor_first else PW-PW//2
    padding = [[0,0],[tmp1,PH-tmp1],[tmp2,PW-tmp2],[0,0]]

    x = np.pad(x, padding, mode='constant', constant_values=0)
    x = x.transpose([3,0,1,2])
    y = np.zeros([H2, W2, C1, N0, H3, W3])
    for ind1 in range(H2):
        for ind2 in range(W2):
            ind3 = slice(ind1, ind1+stride[0]*(H3-1)+1, stride[0])
            ind4 = slice(ind2, ind2+stride[1]*(W3-1)+1, stride[1])
            y[ind1,ind2] = x[:,:,ind3,ind4]
    y = y.reshape([H2*W2*C1,N0*H3*W3])
    dy = dy.reshape([N0*H3*W3, C2])
    return np.dot(y, dy).reshape([H2,W2,C1,C2])


def conv2d_gradx(w, dy, xsize, padding='same', stride=(1,1), floor_first=True):
    """2D convolution gradient wrt. input.

    dy (np,float32,(N0,H3,W3,C2))
    w (np,float32,(H2,W2,C1,C2))
    xsize [H1,W1]
    dx (np,float32,(N0,H1,W1,C1))
    """
    H1,W1 = xsize
    H2,W2,C1,C2 = w.shape
    N0,H3,W3,_ = dy.shape

    if padding == 'same':
        PH = H1 - 1 + H2 - max(H3, H3*stride[0]-1)
        PW = W1 - 1 + W2 - max(W3, W3*stride[1]-1)
    elif padding == 'valid':
        PH = 2*H2 - 2
        PW = 2*W2 - 2
    else:
        PH,PW = padding

    xs = np.zeros([N0,H3,stride[0],W3,stride[1],C2])
    xs[:, :, 0, :, 0, :] = dy
    dy = xs.reshape([N0, H3*stride[0], W3*stride[1], C2])
    dy = dy[:, :H1, :W1, :]

    tmp1 = PH//2 if not floor_first else PH-PH//2 #True for forward, False for backward
    tmp2 = PW//2 if not floor_first else PW-PW//2
    padding = [[0,0],[tmp1,PH-tmp1],[tmp2,PW-tmp2],[0,0]]
    dy = np.pad(dy, padding, mode='constant', constant_values=0)

    dy1 = np.zeros([N0, H1, W1, H2, W2, C2])
    for ind1 in range(H1):
        for ind2 in range(W1):
            dy1[:, ind1, ind2] = dy[:, ind1:ind1+H2, ind2:ind2+W2]
    dy1 = dy1[:,:,:,::-1,::-1,:].reshape([N0*H1*W1, H2*W2*C2])

    w = w.transpose([0,1,3,2]).reshape([H2*W2*C2, C1])
    return np.dot(dy1,w).reshape([N0, H1, W1, C1])


def test(x, w, pad='SAME', stride=(1, 1)):
    y = conv2d(x, w, padding=pad.lower(), stride=stride).ravel()
    xx = tf.constant(x, dtype='float32')
    ww = tf.constant(w, dtype='float32')
    yy = tf.nn.conv2d(xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
    with tf.Session() as sess:
        y_tf = yy.eval().ravel()
    np.testing.assert_almost_equal(y, y_tf, decimal=3)


def test_gradw(x, w, pad='SAME', stride=(1, 1)):
    # [N, H, W, K]
    y = conv2d(x, w, padding=pad.lower(), stride=stride)
    dy = np.random.rand(*y.shape)
    dw = conv2d_gradw(x, dy, ksize=w.shape[:2], padding=pad.lower(), stride=stride)

    # Tensorflow checks
    xx = tf.constant(x, dtype='float32')
    ww = tf.constant(w, dtype='float32')
    dyy = tf.constant(dy, dtype='float32')
    yy = tf.nn.conv2d(
        xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
    dww = tf.squeeze(tf.gradients(yy, ww, dyy), [0])
    with tf.Session() as sess:
        dw_tf = dww.eval()
    np.testing.assert_almost_equal(dw.ravel(), dw_tf.ravel(), decimal=3)


def test_gradx(x, w, pad='SAME', stride=(1, 1)):
    # [N, H, W, K]
    y = conv2d(x, w, padding=pad.lower(), stride=stride)
    dy = np.random.rand(*y.shape)
    dx = conv2d_gradx(w, dy, x.shape[1:3], padding=pad.lower(), stride=stride)

    # Tensorflow checks
    xx = tf.constant(x, dtype='float32')
    ww = tf.constant(w, dtype='float32')
    dyy = tf.constant(dy, dtype='float32')
    yy = tf.nn.conv2d(
        xx, ww, strides=[1, stride[0], stride[1], 1], padding=pad)
    dww = tf.squeeze(tf.gradients(yy, xx, dyy), [0])
    with tf.Session() as sess:
        dx_tf = dww.eval()
    np.testing.assert_almost_equal(dx.ravel(), dx_tf.ravel(), decimal=3)


if __name__ == '__main__':
    # np.random.seed(0)

    for ii in range(5):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(2, 3, 2, 1).astype('float32')
        test(x, w)
        test_gradw(x, w)
        test_gradx(x, w)
        print(ii, 'pass')

    for ii in range(5):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(2, 3, 2, 1).astype('float32')
        test(x, w, pad='VALID')
        test_gradw(x, w, pad='VALID')
        test_gradx(x, w, pad='VALID')
        print(ii, 'pass')

    for ii in range(5):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(2, 3, 2, 1).astype('float32')
        test(x, w, stride=(2, 2))
        test_gradw(x, w, stride=(2, 2))
        test_gradx(x, w, stride=(2, 2))
        print(ii, 'pass')

    for ii in range(5):
        x = np.random.rand(3, 5, 5, 2).astype('float32')
        w = np.random.rand(3, 3, 2, 1).astype('float32')
        test(x, w, pad='VALID', stride=(2, 2))
        test_gradw(x, w, pad='VALID', stride=(2, 2))
        test_gradx(x, w, pad='VALID', stride=(2, 2))
        print(ii, 'pass')