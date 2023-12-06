import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits

import geomstats
import geomstats.geometry.euclidean
import geomstats.geometry.hypersphere
import geomstats.geometry.special_orthogonal
import geomstats.geometry.product_manifold
import geomstats.visualization
import geomstats.learning
import geomstats.learning.online_kmeans
import geomstats.learning.pca


sphere = geomstats.geometry.hypersphere.Hypersphere(dim=2, equip=False)
x0 = sphere.random_uniform(n_samples=2)
fig = plt.figure(figsize=(8,8))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
geomstats.visualization.plot(x0, ax=ax, space="S2", label="Point", s=80)
ax.plot(x0[:, 0], x0[:, 1], x0[:, 2], linestyle="dashed", alpha=0.5)
ax.legend()
fig.savefig('tbd00.png')


sphere = geomstats.geometry.hypersphere.Hypersphere(dim=5)
data = sphere.random_uniform(n_samples=10) #(np,float64,(10,6))
assert np.abs(np.linalg.norm(data, axis=1)-1).max() < 1e-10
clustering = geomstats.learning.online_kmeans.OnlineKMeans(space=sphere, n_clusters=4)
clustering.fit(data)


so3 = geomstats.geometry.special_orthogonal.SpecialOrthogonal(n=3, point_type="vector")
data = so3.random_uniform(n_samples=10) #(np,float64,(10,3))
tpca = geomstats.learning.pca.TangentPCA(space=so3, n_components=2)
tpca.fit(data)
tangent_projected_data = tpca.transform(data) #(np,float64,(10,2))



## plot a grid on H2 with Poincare Disk visualization
# https://github.com/geomstats/geomstats/blob/main/examples/plot_grid_h2.py
space = geomstats.geometry.hyperboloid.Hyperboloid(dim=2)
left = -4.0
right = 4.0
bottom = -4.0
top = 4.0
grid_size = 32
n_steps = 512 #Number of steps along the geodesics defining the grid
tmp0 = [geomstats.backend.array([top, x]) for x in geomstats.backend.linspace(left, right, grid_size)]
tmp1 = [geomstats.backend.array([bottom, x]) for x in geomstats.backend.linspace(left, right, grid_size)]
tmp2 = [geomstats.backend.array([x, left]) for x in geomstats.backend.linspace(top, bottom, grid_size)]
tmp3 = [geomstats.backend.array([x, right]) for x in geomstats.backend.linspace(top, bottom, grid_size)]
starts = [space.from_coordinates(x, "intrinsic") for x in tmp0+tmp2]
ends = [space.from_coordinates(x, "intrinsic") for x in tmp1+tmp3]

fig,ax = plt.subplots()
for start, end in zip(starts, ends):
    geodesic = space.metric.geodesic(initial_point=start, end_point=end)
    tmp0 = geodesic(geomstats.backend.linspace(0.0, 1.0, n_steps))
    geomstats.visualization.plot(tmp0, ax=ax, space="H2_poincare_disk", marker=".", s=1)
fig.tight_layout()


sphere = geomstats.geometry.hypersphere.Hypersphere(dim=2)
sphere.dim #2
x0 = geomstats.backend.array([0, 0, 1]) #north pole
sphere.belongs(x0) #True
sphere.is_tangent(vector=geomstats.backend.array([1, 1, 0]), base_point=x0) #True
sphere.belongs(sphere.random_point()) #True
geomstats.geometry.hypersphere.Hypersphere.random_point(sphere) #same as above


manifold = geomstats.geometry.product_manifold.ProductManifold([
    geomstats.geometry.hypersphere.Hypersphere(dim=2),
    geomstats.geometry.hypersphere.Hypersphere(dim=3),
])
x0 = manifold.random_point() #(np,float64,7)
assert abs(np.linalg.norm(x0[:3])-1) < 1e-10
assert abs(np.linalg.norm(x0[3:])-1) < 1e-10


euclidean = geomstats.geometry.euclidean.Euclidean(dim=2, equip=False)
x0 = euclidean.random_point(n_samples=1000) #uniform [-1,1]
x1 = geomstats.backend.mean(x0, axis=0) #zero mean
