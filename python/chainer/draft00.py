import numpy as np
import chainer as ch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


# chainer.variable.Variable
np0 = np.random.rand(3, 5).astype(np.float32)
ch0 = ch.Variable(np0)
ch0.shape
ch0.dtype #numpy.dtype
ch0.array
# ch0.data #NOT recommended, see https://docs.chainer.org/en/stable/guides/variables.html#variables-and-derivatives


# gradients
np0 = np.random.randn(3,5).astype(np.float32)
ch0 = ch.Variable(np0)
ch1 = ch.functions.sum(ch.functions.sin(ch0))
ch1.backward()
ch0.grad #(np,float32)
ch0.grad_var #(chainer.variable.Variable)
ch0.cleargrad()


# Links
fc0 = ch.links.Linear(5,7)
fc0.W
fc0.b
# fc0.cleargrads()
np0 = np.random.randn(3,5).astype(np.float32)
fc0(np0) #(chainer.varible.Variable)


# custom links
class MyCustomLink00(ch.Link):
    def __init__(self, in_size, out_size):
        super().__init__()
        with self.init_scope():
            self.W = ch.Parameter(ch.initializers.Normal(1/np.sqrt(in_size)), (out_size, in_size))
            self.b = ch.Parameter(0, (out_size,))

    def __call__(self, x):
        ret = ch.functions.matmul(x, self.W.T) + self.b
        return ret
fc0 = MyCustomLink00(5,7)
np0 = np.random.randn(3,5).astype(np.float32)
fc0(np0)


# model
np_data = np.random.rand(233, 3).astype(np.float32)
np_label = np.random.rand(233, 1).astype(np.float32)

class MyChain(ch.Chain):
    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.l1 = ch.links.Linear(3, 5)
            self.l2 = ch.links.Linear(5, 1)

    def forward(self, x):
        h = self.l1(x)
        return self.l2(h)
model = MyChain()
model.cleargrads()
optimizer = ch.optimizers.SGD().setup(model)
# optimizer.add_hook(ch.optimizer_hooks.WeightDecay(0.0005))
loss = ch.functions.mean((model(np_data) - np_label)**2)
loss.backward()
optimizer.update()


# extension
# def hf_learning_rate_decay(trainer):
#     trainer.updater.get_optimizer('main').lr *= 0.1
# trainer.extend(hf_learning_rate_decay, trigger=(10,'epoch'))
