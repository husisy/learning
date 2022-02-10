# 1D ODE problem Dirichlet boundary condition
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# plt.ion()

import deepxde as dde


def hf_pde(x, y):
    dy_xx = dde.grad.hessian(y, x)
    ret = -dy_xx - np.pi ** 2 * tf.sin(np.pi * x)
    return ret

# def hf_boundary(x, on_boundary):
#     return on_boundary

def hf_boundary(x, _):
    return np.isclose(x[0], -1) or np.isclose(x[0], 1)

def hf_fval_boundary(x):
    # ret = np.sin(np.pi * x)
    ret = 0
    return ret

def hf_solution(x):
    ret = np.sin(np.pi * x)
    return ret


if __name__ == "__main__":
    geom = dde.geometry.Interval(-1, 1)
    bc = dde.DirichletBC(geom, hf_fval_boundary, hf_boundary)
    data = dde.data.PDE(geom, hf_pde, bc, num_domain=16, num_boundary=2, solution=hf_solution, num_test=100)

    net = dde.maps.FNN(layer_size=[1,50,50,50,1], activation='tanh', kernel_initializer='Glorot uniform')

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

    checkpointer = dde.callbacks.ModelCheckpoint("model/model.ckpt", save_better_only=True)
    losshistory, train_state = model.train(epochs=10000, callbacks=[checkpointer])

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # Plot PDE residual
    model.restore(f"model/model.ckpt-{train_state.best_step}")
    x = geom.uniform_points(1000, True)
    y = model.predict(x, operator=hf_pde)
    np1 = hf_pde(np)
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x")
    plt.ylabel("PDE residual")
    plt.savefig('tbd00.png')
    # plt.show()
