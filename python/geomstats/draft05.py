# https://geomstats.github.io/notebooks/07_practical_methods__riemannian_kmeans.html
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits

import geomstats
import geomstats.visualization
import geomstats.geometry.hypersphere
import geomstats.geometry.special_orthogonal
import geomstats.learning.kmeans

sphere = geomstats.geometry.hypersphere.Hypersphere(dim=2, equip=False)
cluster = sphere.random_von_mises_fisher(kappa=20, n_samples=140)

SO3 = geomstats.geometry.special_orthogonal.SpecialOrthogonal(3, equip=False)
cluster_1 = cluster @ SO3.random_uniform()
cluster_2 = cluster @ SO3.random_uniform()

sphere_with_metric = geomstats.geometry.hypersphere.Hypersphere(dim=2)
data = geomstats.backend.concatenate((cluster_1, cluster_2), axis=0)
kmeans = geomstats.learning.kmeans.RiemannianKMeans(sphere_with_metric, 2, tol=1e-3)
kmeans.fit(data)

fig = plt.figure(figsize=(10, 10))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
colors = ["red", "blue"]
geomstats.visualization.plot(data, ax=ax, space="S2", marker=".", color="grey")
for i, c in enumerate(kmeans.centroids_):
    geomstats.visualization.plot(data[kmeans.labels_==i], ax=ax, space="S2", marker=".", color=colors[i])
    geomstats.visualization.plot(c, ax=ax, space="S2", marker="*", s=2000, color=colors[i])
ax.set_title("Kmeans on Hypersphere Manifold")
ax.auto_scale_xyz([-1, 1], [-1, 1], [-1, 1])
fig.savefig('tbd00.png')
