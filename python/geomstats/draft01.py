# https://geomstats.github.io/notebooks/21_foundations__sub_riemannian_geometry_and_the_heisenberg_group.html
# Sub-Riemannian geometry on the Heisenberg group
import os
os.environ['GEOMSTATS_BACKEND'] = 'pytorch'
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
import torch

import geomstats
import geomstats.geometry.heisenberg
import geomstats.geometry.sub_riemannian_metric

heis = geomstats.geometry.heisenberg.HeisenbergVectors(equip=False)


def heis_frame(point):
    translations = heis.jacobian_translation(point)
    return translations[..., 0:2]


heis_sr = geomstats.geometry.sub_riemannian_metric.SubRiemannianMetric(heis, frame=heis_frame)

line = np.array([k * np.array([1.0, 0.0, 0.0]) for k in np.linspace(0, 10, 10)])
fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
for i in range(line.shape[0]):
    point = line[i, :]
    frame_at_point = heis_sr.frame(point)
    frame1 = frame_at_point[:, 1]
    frame2 = frame_at_point[:, 0]
    ax.quiver(point[0], point[1], point[2], frame1[0], frame1[1], frame1[2],
        length=0.1, normalize=True, color="orange", alpha=0.8, linestyle="-")
    ax.quiver(point[0], point[1], point[2], frame2[0], frame2[1], frame2[2],
        length=0.1, normalize=True, color="blue", alpha=0.8, linestyle="-")
fig.savefig('tbd00.png')

# pytorch is needed here
base_point = geomstats.backend.array([0.0, 0.0, 0.0])
initial_cotangent = geomstats.backend.array([1.0, 1.0, 1.0])
times = geomstats.backend.linspace(0.0, 10.0, 50)
path = heis_sr.geodesic(initial_point=base_point, initial_cotangent_vec=initial_cotangent, n_steps=1000)
path_1_1 = np.asarray(path(times))

fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(0, 0, 0, marker="+", color="black")
ax.plot3D(path_1_1[:, 0], path_1_1[:, 1], path_1_1[:, 2], "-",
    linewidth=1.5, markersize=0, marker=".", color="green", alpha=0.7)
fig.savefig('tbd00.png')


def heis_frame_riemannian(point, epsilon):
    frame_matrix = heis.jacobian_translation(point)
    scale_matrix = geomstats.backend.array([geomstats.backend.ones((3,)), geomstats.backend.ones((3,)), geomstats.backend.ones((3,)) * epsilon]).T
    return geomstats.backend.einsum("...ij,...ij -> ...ij", frame_matrix, scale_matrix)


hf0 = lambda x: heis_frame_riemannian(x, 1)
heis_epsilon_1 = geomstats.geometry.sub_riemannian_metric.SubRiemannianMetric(heis, frame=hf0)
hf0 = lambda x: heis_frame_riemannian(x, 0)
heis_epsilon_0 = geomstats.geometry.sub_riemannian_metric.SubRiemannianMetric(heis, frame=hf0)

base_point = geomstats.backend.array([0.0, 0.0, 0.0])
initial_cotangent = geomstats.backend.array([1.0, 1.0, 1.0])
times = geomstats.backend.linspace(0.0, 10.0, 50)
path_epsilon_1 = heis_epsilon_1.geodesic(initial_point=base_point, initial_cotangent_vec=initial_cotangent, n_steps=1000)
path_epsilon_0 = heis_epsilon_0.geodesic(initial_point=base_point, initial_cotangent_vec=initial_cotangent, n_steps=1000)
path_1 = np.asarray(path_epsilon_1(times))
path_0 = np.asarray(path_epsilon_0(times))

fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(0, 0, 0, marker="+", color="black")
ax.plot3D(path_1[:, 0], path_1[:, 1], path_1[:, 2], "-",
    linewidth=1.5, markersize=0, marker=".", color="green", alpha=0.7, label="epsilon = 1")
ax.plot3D(path_0[:, 0], path_0[:, 1], path_0[:, 2], "-",
    linewidth=1.5, markersize=0, marker=".", color="red", alpha=0.7, label="epsilon = 0")
ax.legend()
fig.savefig('tbd00.png')


base_point = geomstats.backend.array([0.0, 0.0, 0.0])
initial_cotangent = geomstats.backend.array([1.0, 1.0, 1.0])
times = geomstats.backend.linspace(0.0, 1.0, 50)
path_epsilon_1 = heis_epsilon_1.geodesic(initial_point=base_point, initial_cotangent_vec=initial_cotangent)
path_epsilon_0 = heis_epsilon_0.geodesic(initial_point=base_point, initial_cotangent_vec=initial_cotangent)
path_1 = np.asanyarray(path_epsilon_1(times))
path_0 = np.asanyarray(path_epsilon_0(times))

fig = plt.figure()
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.scatter(0, 0, 0, marker="+", color="black")
ax.plot3D(path_1[:, 0], path_1[:, 1], path_1[:, 2], "-",
    linewidth=1.5, markersize=0, marker=".", color="green", alpha=0.7, label="epsilon = 1")
ax.plot3D(path_0[:, 0], path_0[:, 1], path_0[:, 2], "-",
    linewidth=1.5, markersize=0, marker=".", color="red", alpha=0.7, label="epsilon = 0")
for i in range(path_1.shape[0]):
    if i % 10 == 0:
        point = path_1[i, :]
        frame_at_point = heis_sr.frame(point)
        frame1 = frame_at_point[:, 1]
        frame2 = frame_at_point[:, 0]
        ax.quiver(point[0], point[1], point[2], frame1[0], frame1[1], frame1[2],
            length=1, normalize=True, color="orange", alpha=0.8, linestyle="-")
        ax.quiver(point[0], point[1], point[2], frame2[0], frame2[1], frame2[2],
            length=1, normalize=True, color="blue", alpha=0.8, linestyle="-")
for i in range(path_0.shape[0]):
    if i % 10 == 0:
        point = path_0[i, :]
        frame_at_point = heis_sr.frame(point)
        frame1 = frame_at_point[:, 1]
        frame2 = frame_at_point[:, 0]
        ax.quiver(point[0], point[1], point[2], frame1[0], frame1[1], frame1[2],
            length=1, normalize=True, color="pink", alpha=0.8, linestyle="-")
        ax.quiver(point[0], point[1], point[2], frame2[0], frame2[1], frame2[2],
            length=1, normalize=True, color="yellow", alpha=0.8, linestyle="-")
ax.view_init(3, -96)
ax.legend()
fig.savefig('tbd00.png')
