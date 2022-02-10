import numpy as np

from sklearn.datasets import make_regression
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, RBF

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfe_r5 = lambda x,y: round(hfe(x,y), 5)

def skl_WhiteKernel(N0=401, N1=3):
    np1 = np.random.rand(N0, N1)
    np2 = np.random.rand(N0, N1)
    noise_level = np.random.uniform(0, 1)
    np3 = np.zeros((N0,N0))
    np4 = np.eye(N0)*noise_level
    kernel = WhiteKernel(noise_level)
    np3_ = kernel(np1, np2)
    np4_ = kernel(np1)
    print('skl_WhiteKernel(x,y):: skl vs np: ', hfe_r5(np3_,np3))
    print('skl_WhiteKernel(x):: skl vs np: ', hfe_r5(np4_, np4))


def skl_ExpSineSquared(N0=3, N1=5, length_scale=1, periodicity=3):
    np1 = np.random.rand(N0,N1)
    np2 = np.random.rand(N0,N1)
    tmp1 = np.sqrt(np.sum((np1[:,np.newaxis] - np2[np.newaxis])**2, axis=2))
    tmp2 = np.sin(np.pi / periodicity * tmp1)
    np3 = np.exp(-2 * (tmp2/length_scale)**2)

    kernel = ExpSineSquared(length_scale, periodicity)
    np3_ = kernel(np1, np2)
    print('skl_ExpSineSquared:: np vs skl: ', hfe_r5(np3,np3_))


def skl_RBF(N0=101, N1=3, length_scale=0.6):
    np1 = np.random.rand(N0,N1)
    np2 = np.random.rand(N0,N1)
    tmp1 = np.sum((np1[:,np.newaxis] - np2[np.newaxis])**2, axis=2)
    np3 = np.exp(-tmp1/(2*length_scale**2))

    kernel = RBF(length_scale)
    np3_ = kernel(np1, np2)
    print('skl_RBF:: np vs skl: ', hfe_r5(np3, np3_))


def sklearn_kernel_ridge_regression(num_train=53, num_feature=7, num_test=23):
    x_train, y_train = make_regression(n_samples=num_train, n_features=num_feature)
    x_test = np.random.rand(num_test, num_feature)
    kernel_param = dict(length_scale=4.66, periodicity=12.9)

    krr = KernelRidge(alpha=0.001, kernel=ExpSineSquared(**kernel_param))
    krr.fit(x_train, y_train)

    ret_ = krr.predict(x_test)
    ret = ExpSineSquared(**kernel_param)(x_test, x_train) @ krr.dual_coef_
    print('sklearn_kernel_ridge_regression: np vs sklearn: ', hfe_r5(ret_,ret))


if __name__=='__main__':
    skl_ExpSineSquared()
    print()
    skl_WhiteKernel()
    print()
    skl_RBF()
    print()
    sklearn_kernel_ridge_regression()
