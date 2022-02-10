# Inverse problem for the Lorenz system

import io
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import deepxde as dde

# parameters to be identified
C1 = tf.Variable(1.0)
C2 = tf.Variable(1.0)
C3 = tf.Variable(1.0)

# true values, see p. 15 in https://arxiv.org/abs/1907.04502
C1true = 10
C2true = 15
C3true = 8/3

def Lorenz_system(x, y):
    """
    dy1/dx = 10 * (y2 - y1)
    dy2/dx = y1 * (28 - y3) - y2
    dy3/dx = y1 * y2 - 8/3 * y3
    """
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    dy3_x = dde.grad.jacobian(y, x, i=2)
    ret = [
        dy1_x - C1 * (y2 - y1),
        dy2_x - y1 * (C2 - y3) + y2,
        dy3_x - y1 * y2 + C3 * y3,
    ]
    return ret

def boundary(_, on_initial):
    return on_initial

geom = dde.geometry.TimeDomain(0, 3)

# Initial conditions
ic1 = dde.IC(geom, lambda X: -8, boundary, component=0)
ic2 = dde.IC(geom, lambda X: 7, boundary, component=1)
ic3 = dde.IC(geom, lambda X: 27, boundary, component=2)

# https://github.com/lululxvi/deepxde/raw/master/examples/dataset/Lorenz.npz
tmp0 = np.load('data/Lorenz.npz')
observe_t = tmp0['t']
ob_y = tmp0['y']
observe_y0 = dde.PointSetBC(observe_t, ob_y[:, 0:1], component=0)
observe_y1 = dde.PointSetBC(observe_t, ob_y[:, 1:2], component=1)
observe_y2 = dde.PointSetBC(observe_t, ob_y[:, 2:3], component=2)
plt.plot(observe_t, ob_y)
fig,ax = plt.subplots()
ax.plot(observe_t, ob_y[:,0], label='x')
ax.plot(observe_t, ob_y[:,1], label='y')
ax.plot(observe_t, ob_y[:,2], label='z')
ax.set_xlabel('Time')
ax.legend()
fig.savefig('tbd00.png')
plt.close(fig)


# define data object
data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,
    num_boundary=2,
    anchors=observe_t,
)


# define FNN architecture and compile
net = dde.maps.FNN([1,40,40,40,3], activation='tanh', kernel_initializer='Glorot uniform')
model = dde.Model(data, net)
model.compile("adam", lr=0.001)
fnamevar = "variables.dat"
variable = dde.callbacks.VariableValue([C1,C2,C3],  period=1, filename=fnamevar)
losshistory, train_state = model.train(epochs=60000, callbacks=[variable])

# reopen saved data using callbacks in fnamevar
lines = open(fnamevar, "r").readlines()
# read output data in fnamevar (this line is a long story...)
Chat = np.array([np.fromstring(min(re.findall(re.escape('[')+"(.*?)"+re.escape(']'),line), key=len), sep=',') for line in lines])
l,c = Chat.shape
plt.plot(range(l),Chat[:,0],'r-')
plt.plot(range(l),Chat[:,1],'k-')
plt.plot(range(l),Chat[:,2],'g-')
plt.plot(range(l),np.ones(Chat[:,0].shape)*C1true,'r--')
plt.plot(range(l),np.ones(Chat[:,1].shape)*C2true,'k--')
plt.plot(range(l),np.ones(Chat[:,2].shape)*C3true,'g--')
plt.legend(['C1hat','C2hat','C3hat','True C1','True C2','True C3'],loc = "right")
plt.xlabel('Epoch')
plt.show()

yhat = model.predict(observe_t)
plt.plot(observe_t, ob_y,'-',observe_t, yhat,'--')
plt.xlabel('Time')
plt.legend(['x','y','z','xh','yh','zh'])
plt.title('Training data')
plt.show()
