import numpy as np
import chainer as ch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_chainer_gradient():
    np0 = np.random.randn(3, 5).astype(np.float32)
    np1 = np.random.randn(3, 5).astype(np.float32)
    # np2 = np.sum(np.sin(np0)*np1)
    ret_ = np.cos(np0)*np1

    ch0 = ch.Variable(np0)
    ch2 = ch.functions.sin(ch0)
    ch2.grad = np1
    # ch1 = ch.Variable(np1)
    # ch2 = ch.functions.sum(ch.functions.sin(ch0)*ch1)
    ch2.backward()
    assert hfe(ret_, ch0.grad) < 1e-5

    ch0 = ch.Variable(np0)
    ch1 = ch.Variable(np1)
    ch2 = ch.functions.sum(ch.functions.sin(ch0)*ch1)
    ch2.backward()
    assert hfe(ret_, ch0.grad) < 1e-5


def test_chainer_hessian(N0=3):
    np0 = np.random.rand(N0).astype(np.float32)
    # np1 = np.prod(np0)**2
    hessian_np = 4 * np.prod(np0)**2 / (np0[:,np.newaxis]*np0)
    hessian_np[np.arange(N0), np.arange(N0)] /= 2

    ch0 = ch.Variable(np0)
    ch1 = ch.functions.prod(ch0)**2
    ch1.backward(enable_double_backprop=True)
    gradient = ch0.grad_var
    ret0 = []
    for ind0 in range(N0):
        ch0.cleargrad()
        gradient[ind0].backward(retain_grad=(ind0<(N0-1)))
        ret0.append(ch0.grad.copy())
    ret0 = np.stack(ret0)
    assert hfe(hessian_np, ret0) < 1e-5


def test_chainer_links_linear(N0=3, N1=5, N2=7):
    np0 = np.random.randn(N0, N1).astype(np.float32)
    ch0 = ch.Variable(np0)
    fc0 = ch.links.Linear(N1, N2)
    ret0 = fc0(ch0).array.copy()
    ret_ = np.matmul(np0, fc0.W.array.T) + fc0.b.array
    assert hfe(ret_, ret0) < 1e-5


class MyOperator(ch.FunctionNode):
    # could reduce as forward()
    def forward_cpu(self, inputs):
        x, y = inputs #numpy.ndarray
        self.retain_inputs((0, 1)) #mark x and y as retained
        z = np.exp(x) + np.exp(y)
        return z,

    def forward_gpu(self, inputs):
        x, y = inputs #cupy.ndarray
        self.retain_inputs((0, 1))
        cp = ch.backends.cuda.cupy
        z = cp.exp(x) + cp.exp(y)
        return z,

    def backward(self, target_input_indexes, grad_outputs):
        x, y = self.get_retained_inputs()
        gz, = grad_outputs #unpack, chainer.variable.Variable
        gx = gz * ch.functions.exp(x)
        gy = gz * ch.functions.exp(y)
        return gx, gy

def test_chainer_custom_operator():
    np0 = np.random.randn(3,5).astype(np.float32)
    np1 = np.random.randn(3,5).astype(np.float32)

    ch0 = ch.Variable(np0)
    ch1 = ch.Variable(np1)
    ch2 = ch.functions.sum(ch.functions.exp(ch0) + ch.functions.exp(ch1))
    ch2.backward()
    ret_ = [ch0.grad.copy(), ch1.grad.copy()]

    ch0 = ch.Variable(np0)
    ch1 = ch.Variable(np1)
    ch2 = ch.functions.sum(MyOperator().apply((ch0, ch1))[0])
    ch2.backward()
    ret0 = [ch0.grad.copy(), ch1.grad.copy()]
    assert all(hfe(x,y)<1e-5 for x,y in zip(ret_, ret0))


class DummyModel(ch.Chain):
    def __init__(self, in_size, out_size):
        super().__init__()
        with self.init_scope():
            self.l1 = ch.links.Linear(in_size, 23)
            self.l2 = ch.links.Linear(23, out_size)
    def forward(self, x):
        x = ch.functions.relu(self.l1(x))
        x = self.l2(x)
        return x

def np_softmax(np0):
    tmp0 = np.exp(np0 - np.max(np0, axis=-1, keepdims=True))
    ret = tmp0 / np.sum(tmp0, axis=-1, keepdims=True)
    return ret

def np_cross_entropy(logits, label_int):
    tmp0 = np_softmax(logits)
    ret = - np.mean(np.log(tmp0[np.arange(tmp0.shape[0]), label_int]))
    return ret

def test_chainer_links_Classifier(N0=3, N1=11, N2=10):
    np0 = np.random.randn(N0, N1).astype(np.float32)
    np1 = np.random.randint(N2, size=(N0,))
    model_base = DummyModel(N1, N2)
    model = ch.links.Classifier(model_base)

    ret_ = model(np0, np1)
    ret0 = np_cross_entropy(model_base(np0).array.copy(), np1)
    assert hfe(ret_.array.copy(), ret0) < 1e-5
