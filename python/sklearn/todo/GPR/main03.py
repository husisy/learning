import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

N0 = 20
X = np.random.uniform(0, 5, size=[N0,1])
y = 0.5*np.sin(3*X[:,0]) + np.random.normal(0, 0.5, size=[N0])
X_ = np.linspace(0, 5, 100)[:,np.newaxis]

tmp1 = RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=tmp1, alpha=0)
gp.fit(X,y)
ret1 = gp.predict(X_)
ret2 = np.dot(gp.kernel_(X_, X), gp.alpha_) + gp._y_train_mean

print('GPR predict error: {}'.format(hfe(ret1, ret2)))
