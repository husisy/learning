# Burgers equation
import pickle
import numpy as np
import matplotlib.pyplot as plt
# plt.ion()

import deepxde as dde


def convert_npz_to_pkl(src='Burgers.npz', dst='Burgers.pkl'):
    z0 = np.load(src) #pass on windows
    tmp0 = {
        't': z0['t'], #(np,float64,(100,1))
        'x': z0['x'], #(np,float64,(256,1))
        'usol': z0['usol'], #(np,float64,(256,100))
    }
    with open(dst, 'wb') as fid:
        pickle.dump(tmp0, fid)


def plot_3dsurface(x, t, z):
    assert (x.ndim==1) and (t.ndim==1) and z.shape==(x.shape[0],t.shape[0])
    tt,xx = np.meshgrid(t, x)
    fig,ax = plt.subplots(subplot_kw={'projection':'3d'})
    ax.plot_surface(xx,tt,z)
    ax.set_xlabel('x')
    ax.set_ylabel('t')

def hf_pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

if __name__ == "__main__":
    geom = dde.geometry.Interval(-1, 1)
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    hf0 = lambda x: 0
    hf1 = lambda _, on_boundary: on_boundary
    bc = dde.DirichletBC(geomtime, hf0, hf1)
    hf0 = lambda x: -np.sin(np.pi * x[:, 0:1])
    hf1 = lambda _, on_initial: on_initial
    ic = dde.IC(geomtime, hf0, hf1) #Initial Condition (IC)

    data = dde.data.TimePDE(geomtime, hf_pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160)
    net = dde.maps.FNN([2,20,20,20,1], activation='tanh', kernel_initializer='Glorot normal')
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    model.train(epochs=15000)
    model.compile("L-BFGS-B") #to get a smaller loss
    losshistory, train_state = model.train()
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    # https://github.com/lululxvi/deepxde/blob/master/examples/dataset/Burgers.npz
    # fail to load on linux, but succes on windows, so call convert_npz_to_pkl() first
    with open('data/Burgers.pkl', 'rb') as fid:
        z0 = pickle.load(fid)
    xx, tt = np.meshgrid(z0['x'], z0['t']) #(np,float64,(100,256))
    X = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1)
    y_true = z0['usol'].T.reshape(-1,1)
    y_pred = model.predict(X)
    f = model.predict(X, operator=hf_pde)
    print("Mean residual:", np.mean(np.absolute(f)))
    print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))

    # plot_3dsurface(z0['x'][:,0], z0['t'][:,0], z0['usol'])
    # tmp0 = y_pred.reshape(z0['usol'].shape[::-1]).T
    # plot_3dsurface(z0['x'][:,0], z0['t'][:,0], tmp0)
