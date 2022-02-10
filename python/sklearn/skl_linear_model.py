import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfe_r5 = lambda x,y,eps=1e-5: round(hfe(x,y,eps),5)

def skl_linear_regression():
    tmp1 = load_boston()
    np1 = tmp1['data']
    np2 = tmp1['target']
    regr = LinearRegression()

    regr.fit(np1, np2)
    np3 = regr.predict(np1)
    np3_ = np.matmul(np1, regr.coef_) + regr.intercept_
    print('skl_linear_regression:: np vs skl: ', hfe_r5(np3,np3_))


def skl_ridge(N0=8, N1=9, noise=0.3, alpha=np.logspace(-10,-2,100), use_meaningless_data=True):
    '''http://scikit-learn.org/stable/auto_examples/linear_model/plot_ridge_path.html'''
    if use_meaningless_data:
        print('WARNING! the used data CANNOT explain the effect of alpha.')
        npx = 1 / (np.arange(0,N0)[:,np.newaxis] + np.arange(1,N1))
        npy = np.ones(N0)
    else:
        tmp1 = np.random.normal(0, 1, size=[N0,N1])
        tmp2 = np.random.normal(N1)
        npx = tmp1 + np.random.normal(0, noise/2, size=[N0,N1])
        npy = np.sum(tmp1*tmp2, axis=1)

    coefs = np.zeros([alpha.shape[0], npx.shape[1]])
    for ind1,a in enumerate(alpha):
        ridge = Ridge(alpha=a, fit_intercept=False)
        ridge.fit(npx, npy)
        coefs[ind1] = ridge.coef_

    fig = plt.figure()
    ax = fig.add_axes([0.15,0.15,0.7,0.7])
    ax.plot(alpha, coefs)
    ax.set(xscale='log', xlabel='alpha', ylabel='weights', title='skl_ridge')


def skl_logistic_regression():
    # http://scikit-learn.org/stable/auto_examples/linear_model/plot_logistic.html#sphx-glr-auto-examples-linear-model-plot-logistic-py
    raise Exception('not implement yet')


# TODO # ref: https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html#sphx-glr-auto-examples-applications-plot-tomography-l1-reconstruction-py

if __name__=='__main__':
    skl_linear_regression()
    print()
    skl_ridge()
