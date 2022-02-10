import numpy as np
import tensorflow as tf
# reference: https://github.com/renmengye/np-conv2d
# TODO change to np.lib.stride_tricks.as_strided for efficiency
# see np_vs_xx/tensorflow/zcprivate/numpyflow/conv2d.py

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-5))
hf_randc = lambda *size: np.random.randn(*size) + np.random.randn(*size) * 1j

def conv2d(x, w, padding='same', stride=(1,1), floor_first=True):
    """2D convolution (technically speaking, correlation).
    https://github.com/renmengye/np-conv2d

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

    y = np.zeros([N0, H3, W3, H2, W2, C1], dtype=x.dtype)
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


def hf_test(x, w, padding, grad_y=None, stride=(1,1)):
    y = conv2d(x, w, padding=padding, stride=stride)
    if grad_y is None:
        grad_y = np.random.rand(*y.shape).astype(np.float32)
    grad_x = conv2d_gradx(w, grad_y, x.shape[1:3], padding=padding, stride=stride)
    grad_w = conv2d_gradw(x, grad_y, ksize=w.shape[:2], padding=padding, stride=stride)

    tf_x = tf.convert_to_tensor(x)
    tf_w = tf.convert_to_tensor(w)
    with tf.GradientTape() as tape:
        tape.watch([tf_x, tf_w])
        tf_y = tf.nn.conv2d(tf_x, tf_w, strides=[1,stride[0],stride[1],1], padding=padding.upper())
    tf_grad_x,tf_grad_w = tape.gradient(tf_y, [tf_x,tf_w], [grad_y])
    assert hfe(y, tf_y.numpy()) < 1e-5
    assert hfe(grad_x, tf_grad_x.numpy()) < 1e-5
    assert hfe(grad_w, tf_grad_w.numpy()) < 1e-5


def test_conv2d_padding_same():
    x = np.random.rand(3, 5, 5, 2).astype(np.float32)
    w = np.random.rand(2, 3, 2, 1).astype(np.float32)
    hf_test(x, w, 'same')


def test_conv2d_padding_valid():
    x = np.random.rand(3, 5, 5, 2).astype(np.float32)
    w = np.random.rand(2, 3, 2, 1).astype(np.float32)
    hf_test(x, w, 'valid')


def test_conv2d_same_stride():
    x = np.random.rand(3, 5, 5, 2).astype(np.float32)
    w = np.random.rand(2, 3, 2, 1).astype(np.float32)
    hf_test(x, w, 'same', stride=(2, 2))


def test_conv2d_valid_stride():
    x = np.random.rand(3, 5, 5, 2).astype(np.float32)
    w = np.random.rand(3, 3, 2, 1).astype(np.float32)
    hf_test(x, w, 'valid', stride=(2, 2))


class MyComplexConv2d(tf.Module):
    def __init__(self, filters, kernel_size, padding='same', strides=(1,1)):
        super().__init__()
        assert isinstance(filters, int) and filters>0
        assert isinstance(kernel_size,tuple) and len(kernel_size)==2
        assert isinstance(kernel_size[0],int) and kernel_size[0]>0
        assert isinstance(kernel_size[1],int) and kernel_size[1]>0
        assert isinstance(padding, str)
        padding = padding.lower()
        assert padding in {'same', 'valid'}
        assert isinstance(strides,tuple) and len(strides)==2
        assert isinstance(strides[0],int) and strides[0]>0
        assert isinstance(strides[1],int) and strides[1]>0
        self.conv_real = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.conv_imag = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False)
        self.bias_real = tf.Variable(np.zeros(filters), dtype=tf.float32)
        self.bias_imag = tf.Variable(np.zeros(filters), dtype=tf.float32)

    def __call__(self, x):
        x_real = tf.math.real(x)
        x_imag = tf.math.imag(x)
        tmp0 = self.conv_real(x_real) - self.conv_imag(x_imag) + self.bias_real
        tmp1 = self.conv_real(x_imag) + self.conv_imag(x_real) + self.bias_imag
        return tf.dtypes.complex(tmp0, tmp1)


def test_nn_conv2d_complex():
    npx = hf_randc(3,7,11,2)
    tfx = tf.convert_to_tensor(npx, dtype=tf.complex64)

    cconv2d = MyComplexConv2d(3, (2,5), padding='same', strides=(1,1))
    tf0 = cconv2d(tfx)
    npw = cconv2d.conv_real.kernel.numpy() + 1j*cconv2d.conv_imag.kernel.numpy() #(np,complex,(2,5,2,3))
    npb = cconv2d.bias_real.numpy() + 1j*cconv2d.bias_imag.numpy()
    np0 = conv2d(npx, npw, padding='same', stride=(1,1)) + npb
    assert hfe(np0, tf0.numpy()) < 1e-5
