import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.scipy

import pymanopt


N = 1000 # Number of data points
D = 2 # Dimension of each data point
K = 3 # Number of clusters

pi = [0.1, 0.6, 0.3]
mu = [np.array([-4, 1]), np.array([0, 0]), np.array([2, -1])]
Sigma = [
    np.array([[3, 0], [0, 1]]),
    np.array([[1, 1.0], [1, 3]]),
    0.5 * np.eye(2),
]

components = np.random.choice(K, size=N, p=pi)
samples = np.zeros((N, D))
for k in range(K):
    indices = k == components
    n_k = indices.sum()
    if n_k > 0:
        samples[indices] = np.random.multivariate_normal(mu[k], Sigma[k], n_k)

colors = ["r", "g", "b", "c", "m"]
fig,ax = plt.subplots()
for k in range(K):
    indices = k == components
    ax.scatter(samples[indices, 0], samples[indices, 1], alpha=0.4, color=colors[k % K])
ax.axis("equal")
fig.tight_layout()
fig.savefig('tbd00.png', dpi=200)



manifold = pymanopt.manifolds.Product([pymanopt.manifolds.SymmetricPositiveDefinite(D + 1, k=K), pymanopt.manifolds.Euclidean(K - 1)])

@pymanopt.function.autograd(manifold)
def cost(S, v):
    nu = np.append(v, 0)
    logdetS = np.expand_dims(np.linalg.slogdet(S)[1], 1)
    y = np.concatenate([samples.T, np.ones((1, N))], axis=0)

    # Calculate log_q
    y = np.expand_dims(y, 0)

    # 'Probability' of y belonging to each cluster
    log_q = -0.5 * (np.sum(y * np.linalg.solve(S, y), axis=1) + logdetS)

    alpha = np.exp(nu)
    alpha = alpha / np.sum(alpha)
    alpha = np.expand_dims(alpha, 1)

    loglikvec = autograd.scipy.special.logsumexp(np.log(alpha) + log_q, axis=0)
    return -np.sum(loglikvec)


problem = pymanopt.Problem(manifold, cost)
optimizer = pymanopt.optimizers.SteepestDescent(verbosity=1)
result = optimizer.run(problem) #min step_size reached after 251 iterations, 54.13 seconds.
Xopt = result.point

mu1hat = Xopt[0][0][0:2, 2:3]
Sigma1hat = Xopt[0][0][:2, :2] - mu1hat @ mu1hat.T
mu2hat = Xopt[0][1][0:2, 2:3]
Sigma2hat = Xopt[0][1][:2, :2] - mu2hat @ mu2hat.T
mu3hat = Xopt[0][2][0:2, 2:3]
Sigma3hat = Xopt[0][2][:2, :2] - mu3hat @ mu3hat.T
pihat = np.exp(np.concatenate([Xopt[1], [0]], axis=0))
pihat = pihat / np.sum(pihat)
