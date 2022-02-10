import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import sklearn.manifold
import sklearn.decomposition
import sklearn.datasets

n_points = 1000
X, color = sklearn.datasets.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

hf0 = lambda x: sklearn.manifold.LocallyLinearEmbedding(method=x,
        n_neighbors=n_neighbors, n_components=n_components, eigen_solver='auto')
methods = dict()
methods['LLE'] = hf0('standard').fit_transform(X)
methods['LTSA'] = hf0('ltsa').fit_transform(X)
methods['Hessian LLE'] = hf0('hessian').fit_transform(X)
# methods['Modified LLE'] = hf0('modified').fit_transform(X)
methods['PCA'] = sklearn.decomposition.PCA(n_components=n_components).fit_transform(X)
methods['Isomap'] = sklearn.manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components).fit_transform(X)
methods['MDS'] = sklearn.manifold.MDS(n_components=n_components, max_iter=100, n_init=1).fit_transform(X)
methods['SE'] = sklearn.manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors).fit_transform(X)
methods['t-SNE'] = sklearn.manifold.TSNE(n_components=n_components, init='pca', random_state=0).fit_transform(X)

# Plot results
# Create figure
fig = plt.figure(figsize=(15, 8))
fig.suptitle("Manifold Learning with %i points, %i neighbors" % (1000, n_neighbors), fontsize=14)

# Add 3d scatter plot
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
ax.view_init(4, -72)

for i, (label, Y) in enumerate(methods.items()):
    ax = fig.add_subplot(2, 5, 2 + i + (i > 3))
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('tight')


import numpy as np
hf0 = lambda x: x+x
N0 = 5
np_rng = np.random.default_rng()
tmp0 = np.linalg.svd(np_rng.normal(size=(N0,N0)))[0]
tmp1 = np_rng.normal(size=N0)
np0 = tmp0 @ np.diag(tmp1-tmp1.min()) @ tmp0.T
np.linalg.eigvalsh(np0)

np1 = np_rng.normal(size=(N0,N0-1))
np2 =  np1.T @ np0 @ np1
np.linalg.eigvalsh(np2)


import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sklearn.manifold
import sklearn.datasets

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


## SE
n_point = 1000
X, color = sklearn.datasets.make_s_curve(n_point, random_state=0)
n_neighbors = 10
n_components = 2
random_state = 233
model_se = sklearn.manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
adjacency_sparse = model_se._get_affinity_matrix(X)
ret_ = sklearn.manifold._spectral_embedding.spectral_embedding(adjacency_sparse,
    n_components=n_components, eigen_solver=None, random_state=random_state)
degree_sparse = scipy.sparse.spdiags(np.asarray(adjacency_sparse.sum(axis=1))[:,0], 0, *adjacency_sparse.shape)
laplacian_sparse = degree_sparse - adjacency_sparse
EVL,EVC = scipy.sparse.linalg.eigsh(laplacian_sparse, M=degree_sparse,
        k=n_components+1, which='SA', return_eigenvectors=True)
EVC = EVC[:,1:]
tmp0 = (ret_*EVC).sum(axis=0) / (np.sqrt((ret_*ret_).sum(axis=0)) * np.sqrt((EVC*EVC).sum(axis=0)))
assert np.all(np.abs(np.abs(tmp0)-1) < 1e-5)


## AGH (brute force)
import sklearn.cluster
n_point = 1000
n_anchor = 50
num_nn = 10
AGH_t = 1
n_component = 2
X, color = sklearn.datasets.make_s_curve(n_point, random_state=0)

model_kmeans = sklearn.cluster.KMeans(n_clusters=n_anchor)
model_kmeans.fit(X)
x_anchor = model_kmeans.cluster_centers_
distance = np.linalg.norm(X[:,np.newaxis] - x_anchor, ord=2, axis=2)
tmp0 = np.sort(distance, axis=1)[:,:num_nn]
tmp1 = np.exp(-tmp0**2/AGH_t)
tmp2 = (np.arange(n_point)[:,np.newaxis]*n_anchor + np.argsort(distance, axis=1)[:,:num_nn]).reshape(-1)
AGH_Z = np.zeros((n_point,n_anchor), dtype=np.float64)
AGH_Z.reshape(-1)[tmp2] = (tmp1/tmp1.sum(axis=1, keepdims=True)).reshape(-1)
adjacency_np = (AGH_Z/AGH_Z.sum(axis=0)) @ AGH_Z.T
tmp0 = AGH_Z/np.sqrt(AGH_Z.sum(axis=0))
EVL,EVC = np.linalg.eigh(tmp0.T @ tmp0)
EVL = EVL[-2::-1][:n_component]
EVC = EVC[:,-2::-1][:,:n_component]
AGH_inv_sqrt_D = 1/np.sqrt(AGH_Z.sum(axis=0))
AGH_Y = np.sqrt(n_point) * (AGH_Z * AGH_inv_sqrt_D) @ (EVC/np.sqrt(EVL))
assert np.all(np.abs(AGH_Y.sum(axis=0))<1e-5)
assert hfe((AGH_Y.T @ AGH_Y)/n_point, np.eye(n_component)) < 1e-5
fig,ax = plt.subplots()
ax.scatter(AGH_Y[:, 0], AGH_Y[:, 1], c=color, cmap=plt.cm.Spectral)


## AGH(digits)
import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import sklearn.manifold
import sklearn.datasets
import sklearn.cluster

import matplotlib.pyplot as plt
import matplotlib.offsetbox
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(y[i]), color=plt.cm.Set1(y[i] / 10.), fontdict={'weight': 'bold', 'size': 9})

    # only print thumbnails with matplotlib > 1.0
    shown_images = np.array([[1., 1.]])  # just something big
    for i in range(X.shape[0]):
        dist = np.sum((X[i] - shown_images) ** 2, 1)
        if np.min(dist) < 4e-3:
            # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [X[i]]]
        imagebox = matplotlib.offsetbox.AnnotationBbox(matplotlib.offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r), X[i])
        ax.add_artist(imagebox)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

digits = sklearn.datasets.load_digits(n_class=6)
X = digits.data
y = digits.target
n_point = X.shape[0]
# n_samples, n_features = X.shape
# n_neighbors = 30
n_anchor = 100
num_nn = 10
AGH_t = 2
n_component = 2

model_kmeans = sklearn.cluster.KMeans(n_clusters=n_anchor)
model_kmeans.fit(X)
x_anchor = model_kmeans.cluster_centers_
distance = np.linalg.norm(X[:,np.newaxis] - x_anchor, ord=2, axis=2)
tmp0 = np.sort(distance, axis=1)[:,:num_nn]
tmp1 = np.exp(-tmp0**2/AGH_t)
tmp2 = (np.arange(n_point)[:,np.newaxis]*n_anchor + np.argsort(distance, axis=1)[:,:num_nn]).reshape(-1)
AGH_Z = np.zeros((n_point,n_anchor), dtype=np.float64)
AGH_Z.reshape(-1)[tmp2] = (tmp1/tmp1.sum(axis=1, keepdims=True)).reshape(-1)
adjacency_np = (AGH_Z/AGH_Z.sum(axis=0)) @ AGH_Z.T
tmp0 = AGH_Z/np.sqrt(AGH_Z.sum(axis=0))
EVL,EVC = np.linalg.eigh(tmp0.T @ tmp0)
EVL = EVL[-2::-1][:n_component]
EVC = EVC[:,-2::-1][:,:n_component]
AGH_inv_sqrt_D = 1/np.sqrt(AGH_Z.sum(axis=0))
AGH_Y = np.sqrt(n_point) * (AGH_Z * AGH_inv_sqrt_D) @ (EVC/np.sqrt(EVL))
# plot_embedding(AGH_Y, title='AGH')

fig,ax = plt.subplots()
ax.scatter(AGH_Y[:, 0], AGH_Y[:, 1], cmap=plt.cm.Spectral)
