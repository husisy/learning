# https://github.com/lululxvi/deepxde/blob/master/examples/ode_system.py
import numpy as np

import deepxde as dde

def ode_system(x, y):
    """
    dy1/dx = y2
    dy2/dx = -y1
    """
    # x(tf.Placeholder,float32,(?,1))
    # y(tf,float32,(?,2))
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)
    dy2_x = dde.grad.jacobian(y, x, i=1)
    return [dy1_x - y2, dy2_x + y1]

def boundary(x, on_initial):
    # x(np,float32,(1,))
    # on_initial(bool)
    return on_initial

def hf_solution(x):
    # x(np,float32,(?,1))
    return np.hstack((np.sin(x), np.cos(x)))


if __name__ == "__main__":
    geom = dde.geometry.TimeDomain(0, 10)
    ic1 = dde.IC(geom, np.sin, boundary, component=0)
    ic2 = dde.IC(geom, np.cos, boundary, component=1)
    data = dde.data.PDE(geom, ode_system, [ic1, ic2], 35, 2, solution=hf_solution, num_test=100)
    net = dde.maps.FNN([1,50,50,50,2], activation='tanh', kernel_initializer='Glorot uniform')

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    # dde.saveplot(losshistory, train_state, issave=True, isplot=True)
