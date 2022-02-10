import numpy as np
import mxnet as mx

## ndarray
mx.nd.array([[2,3,3],[2,23,233]])
mx.nd.ones((2,3))
mx.nd.full((2,3), 0.233)
mx0 = mx.nd.uniform(-1, 1, (2,3))
mx0.shape
mx0.size
mx0.dtype
mx0.T
mx0.asnumpy()
mx1 = mx.nd.array(mx0.asnumpy())

mx0 = mx.nd.uniform(-1, 1, (2,3))
mx1 = mx.nd.uniform(-1, 1, (2,3))
mx0*mx1
mx0.exp()
mx.nd.dot(mx0, mx0.T)

mx0 = mx.nd.uniform(-1, 1, (2,3))
mx0[1,2]
mx0[:,1:3]


## neural network
z0 = mx.gluon.nn.Dense(7)
z0.initialize()
mx0 = mx.nd.uniform(-1, 1, (3,5))
mx1 = z0(mx0) #(mx,float32,(3,7))
z0.weight.data()
z0.bias.data()

net = mx.gluon.nn.Sequential()
net.add(
    mx.gluon.nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
    mx.gluon.nn.MaxPool2D(pool_size=2, strides=2),
    mx.gluon.nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
    mx.gluon.nn.MaxPool2D(pool_size=2, strides=2),
    mx.gluon.nn.Dense(120, activation="relu"),
    mx.gluon.nn.Dense(84, activation="relu"),
    mx.gluon.nn.Dense(10),
)
net.initialize()
mx0 = mx.nd.random.uniform(shape=(5,1,28,28))
mx1 = net(mx0) #(mx,float32,(5,10))
net[0].bias.data()


class MixMLP(mx.gluon.nn.Block):
    def __init__(self):
        super(MixMLP, self).__init__()
        self.fc0 = mx.gluon.nn.Dense(3, activation='relu')
        self.fc1 = mx.gluon.nn.Dense(4)
        self.fc2 = mx.gluon.nn.Dense(5)
    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        x = mx.nd.relu(x)
        x = self.fc2(x)
        return x
net = MixMLP()
net.initialize()
mx0 = mx.nd.random.uniform(shape=(5,2))
mx1 = net(mx0) #(mx,float32,(5,5))


## autograd
mx0 = mx.nd.random.uniform(shape=(2,3))
mx0.attach_grad()
with mx.autograd.record():
    mx1 = 2*mx0**2
mx1.backward()
mx0.grad

## gpu
device_gpu = mx.gpu(0)
mx0 = mx.nd.ones((3,4), ctx=device_gpu)
mx0 = mx.nd.random.uniform(shape=(3,4), ctx=device_gpu)


## state https://gluon.mxnet.io/chapter03_deep-neural-networks/mlp-dropout-gluon.html#Accessing-is_training()-status
with mx.autograd.predict_mode():
    print(mx.autograd.is_training())
with mx.autograd.train_mode():
    print(mx.autograd.is_training())
