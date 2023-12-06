# Fréchet Mean and Tangent PCA
# https://geomstats.github.io/notebooks/06_practical_methods__riemannian_frechet_mean_and_tangent_pca.html
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits

import geomstats
import geomstats.visualization
import geomstats.learning.frechet_mean
import geomstats.learning.pca
import geomstats.geometry.hypersphere
import geomstats.geometry.hyperboloid

## sphere
sphere = geomstats.geometry.hypersphere.Hypersphere(dim=2)
data = sphere.random_von_mises_fisher(kappa=15, n_samples=140)

mean = geomstats.learning.frechet_mean.FrechetMean(sphere)
mean.fit(data)

tpca = geomstats.learning.pca.TangentPCA(sphere, n_components=2)
tpca.fit(data, base_point=mean.estimate_)
tangent_projected_data = tpca.transform(data)
tpca.explained_variance_ratio_ # [0.54461944, 0.45538056]
geodesic_0 = sphere.metric.geodesic(initial_point=mean.estimate_, initial_tangent_vec=tpca.components_[0])
geodesic_1 = sphere.metric.geodesic(initial_point=mean.estimate_, initial_tangent_vec=tpca.components_[1])

t = geomstats.backend.linspace(-1.0, 1.0, 100)
geodesic_points_0 = geodesic_0(t)
geodesic_points_1 = geodesic_1(t)

fig = plt.figure(figsize=(10,10))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax = geomstats.visualization.plot(geodesic_points_0, ax, space="S2", linewidth=2, label="First component")
ax = geomstats.visualization.plot(geodesic_points_1, ax, space="S2", linewidth=2, label="Second component")
ax = geomstats.visualization.plot(data, ax, space="S2", color="black", alpha=0.2, label="Data points")
ax = geomstats.visualization.plot(mean.estimate_, ax, space="S2", color="red", s=200, label="Fréchet mean")
ax.legend()
ax.set_box_aspect([1, 1, 1])
fig.savefig('tbd00.png')


## hyperbolic
hyperbolic_plane = geomstats.geometry.hyperboloid.Hyperboloid(dim=2)
data = hyperbolic_plane.random_point(n_samples=140)

mean = geomstats.learning.frechet_mean.FrechetMean(hyperbolic_plane)
mean.fit(data)

tpca = geomstats.learning.pca.TangentPCA(hyperbolic_plane, n_components=2)
tpca.fit(data, base_point=mean.estimate_)
tangent_projected_data = tpca.transform(data)
tpca.explained_variance_ratio_
geodesic_0 = hyperbolic_plane.metric.geodesic(initial_point=mean.estimate_, initial_tangent_vec=tpca.components_[0])
geodesic_1 = hyperbolic_plane.metric.geodesic(initial_point=mean.estimate_, initial_tangent_vec=tpca.components_[1])
t = geomstats.backend.linspace(-1.0, 1.0, 100)
geodesic_points_0 = geodesic_0(t)
geodesic_points_1 = geodesic_1(t)

fig,ax = plt.subplots(figsize=(8,8))
ax = geomstats.visualization.plot(geodesic_points_0, ax, space="H2_poincare_disk", linewidth=2, label="First component")
ax = geomstats.visualization.plot( geodesic_points_1, ax, space="H2_poincare_disk", linewidth=2, label="Second component")
ax = geomstats.visualization.plot(data, ax, space="H2_poincare_disk", color="black", alpha=0.2, label="Data points")
ax = geomstats.visualization.plot(mean.estimate_, ax, space="H2_poincare_disk", color="red", s=200, label="Fréchet mean")
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png')
