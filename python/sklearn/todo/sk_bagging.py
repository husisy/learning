import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# parameter
n_repeat = 50
n_train = 50
n_test = 1000
noise = 0.1
# np.random.seed(0)

# generate data
hf1 = lambda x: np.exp(-x**2) + 1.5*np.exp(-(x-2)**2)
x_train = [np.sort(np.random.rand(n_train)*10-5) for _ in range(n_repeat)]
y_train = [hf1(x)+np.random.normal(0,noise,x.shape) for x in x_train]
x_test = np.sort(np.random.rand(n_test)*10-5)
y_test = hf1(x_test)[:,np.newaxis] + np.random.normal(0,noise,(n_test,n_repeat))

# estimator
plt.figure(figsize=(10, 8))
estimators = [("Tree", DecisionTreeRegressor(max_depth=3)),
              ("Bagging(Tree)", BaggingRegressor(DecisionTreeRegressor(max_depth=3), n_estimators=10))]
n_estimators = len(estimators)
for ind1, (name, estimator) in enumerate(estimators):
    y_predict = np.zeros((n_test, n_repeat))

    for ind2 in range(n_repeat):
        estimator.fit(x_train[ind2][:,np.newaxis], y_train[ind2])
        y_predict[:, ind2] = estimator.predict(x_test[:,np.newaxis])

    # Bias^2 + Variance + Noise decomposition of the mean squared error
    y_error = np.mean((y_predict[:,np.newaxis]-y_test[:,:,np.newaxis])**2, axis=(1,2))
    y_noise = np.var(y_test, axis=1)
    y_bias = (hf1(x_test) - np.mean(y_predict, axis=1))**2
    y_var = np.var(y_predict, axis=1)

    tmp1 = '{0}:: error/bias^2/var/noise: {1:.4f}/{2:.4f}/{3:.4f}/{4:.4f}'
    print(tmp1.format(name, np.mean(y_error), np.mean(y_bias), np.mean(y_var), np.mean(y_noise)))

    plt.subplot(2, n_estimators, ind1+1)
    plt.plot(x_test, hf1(x_test), "b", label="$f(x)$")
    plt.plot(x_train[0], y_train[0], ".b", label="LS ~ $y = f(x)+noise$")

    for ind2 in range(n_repeat):
        if ind2 == 0:
            plt.plot(x_test, y_predict[:,ind2], "r", label="$\^y(x)$")
        else:
            plt.plot(x_test, y_predict[:,ind2], "r", alpha=0.05)

    plt.plot(x_test, np.mean(y_predict, axis=1), "c", label="$\mathbb{E}_{LS} \^y(x)$")

    plt.xlim([-5, 5])
    plt.title(name)

    if ind1 == n_estimators-1: plt.legend(loc=(1.1,0.5))

    plt.subplot(2, n_estimators, n_estimators+ind1+1)
    plt.plot(x_test, y_error, "r", label="$error(x)$")
    plt.plot(x_test, y_bias, "b", label="$bias^2(x)$"),
    plt.plot(x_test, y_var, "g", label="$variance(x)$"),
    plt.plot(x_test, y_noise, "c", label="$noise(x)$")

    plt.xlim([-5, 5])
    plt.ylim([0, 0.1])

    if ind1 == n_estimators-1: plt.legend(loc=(1.1,0.5))

plt.subplots_adjust(right=.75)
plt.show()
