# https://geomstats.github.io/notebooks/04_practical_methods__from_vector_spaces_to_manifolds.html
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits

import geomstats
import geomstats.datasets.utils
import geomstats.geometry.hypersphere
import geomstats.geometry.hyperboloid
import geomstats.geometry.special_euclidean
import geomstats.visualization

data, names = geomstats.datasets.utils.load_cities()
sphere = geomstats.geometry.hypersphere.Hypersphere(dim=2)

# exponential map
paris = data[19]
vector = geomstats.backend.array([1, 0, 0.8])
tangent_vector = sphere.to_tangent(vector, base_point=paris)
result = sphere.metric.exp(tangent_vector, base_point=paris)
geodesic = sphere.metric.geodesic(initial_point=paris, initial_tangent_vec=tangent_vector)
points_on_geodesic = geodesic(geomstats.backend.linspace(0.0, 1.0, 30))
fig = plt.figure(figsize=(10,10))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
geomstats.visualization.plot(paris, ax=ax, space="S2", s=100, alpha=0.8, label="Paris")
geomstats.visualization.plot(result, ax=ax, space="S2", s=100, alpha=0.8, label="End point")
geomstats.visualization.plot(points_on_geodesic, ax=ax, space="S2", color="black", label="Geodesic")
arrow = geomstats.visualization.Arrow3D(paris, vector=tangent_vector)
arrow.draw(ax, color="black")
ax.legend()
fig.savefig('tbd00.png')


# logarithmic map
paris = data[19]
beijing = data[15]
log = sphere.metric.log(point=beijing, base_point=paris)
geodesic_func = sphere.metric.geodesic(initial_point=paris, end_point=beijing)
points_on_geodesic = geodesic_func(geomstats.backend.linspace(0.0, 1.0, 30))
fig = plt.figure(figsize=(10,10))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
geomstats.visualization.plot(paris, ax=ax, space="S2", s=100, alpha=0.8, label="Paris")
geomstats.visualization.plot(beijing, ax=ax, space="S2", s=100, alpha=0.8, label="Beijing")
geomstats.visualization.plot(points_on_geodesic, ax=ax, space="S2", color="black", label="Geodesic")
arrow = geomstats.visualization.Arrow3D(paris, vector=log)
arrow.draw(ax, color="black")
ax.legend()
fig.savefig('tbd00.png')


## hyperbolic geodesic
hyperbolic = geomstats.geometry.hyperboloid.Hyperboloid(dim=2)
initial_point = geomstats.backend.array([geomstats.backend.sqrt(2.0), 1.0, 0.0])
end_point = hyperbolic.from_coordinates(geomstats.backend.array([2.5, 2.5]), "intrinsic")
geodesic_func = hyperbolic.metric.geodesic(initial_point=initial_point, end_point=end_point)
points = geodesic_func(geomstats.backend.linspace(0.0, 1.0, 10))

fig,ax = plt.subplots(figsize=(8,8))
geomstats.visualization.plot(initial_point, ax=ax, space="H2_poincare_disk", s=50, label="Initial point")
geomstats.visualization.plot(end_point, ax=ax, space="H2_poincare_disk", s=50, label="End point")
geomstats.visualization.plot(points[1:-1], ax=ax, space="H2_poincare_disk", s=5, color="black", label="Geodesic")
ax.set_title("Geodesic on the hyperbolic plane in Poincare disk representation")
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png')

fig,ax = plt.subplots(figsize=(8,8))
ax = geomstats.visualization.plot(initial_point, ax=ax, space="H2_klein_disk", s=50, label="Initial point")
ax = geomstats.visualization.plot(end_point, ax=ax, space="H2_klein_disk", s=50, label="End point")
ax = geomstats.visualization.plot(points[1:-1], ax=ax, space="H2_klein_disk", s=5, color="black", label="Geodesic")
ax.set_title("Geodesic on the hyperbolic plane in Klein disk representation")
ax.legend()
fig.tight_layout()
fig.savefig('tbd00.png')


## Special Euclidean SE(3)
se3 = geomstats.geometry.special_euclidean.SpecialEuclidean(n=3, point_type="vector")
initial_point = se3.identity
initial_tangent_vec = geomstats.backend.array([1.8, 0.2, 0.3, 3.0, 3.0, 1.0])
geodesic = se3.metric.geodesic(initial_point=initial_point, initial_tangent_vec=initial_tangent_vec)
points = geodesic(geomstats.backend.linspace(-3.0, 3.0, 40))
fig = plt.figure(figsize=(8,8))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
geomstats.visualization.plot(points, ax=ax, space="SE3_GROUP")
fig.savefig('tbd00.png')
