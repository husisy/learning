import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits

import geomstats
import geomstats.geometry.euclidean
import geomstats.geometry.hypersphere
import geomstats.geometry.special_orthogonal
import geomstats.geometry.spd_matrices
import geomstats.visualization
import geomstats.datasets.utils


data, names = geomstats.datasets.utils.load_cities()
names[:5] #['Tokyo', 'New York', 'Mexico City', 'Mumbai', 'São Paulo']
# data (np,float64,(50,3))
sphere = geomstats.geometry.hypersphere.Hypersphere(dim=2, equip=False)
assert np.all(sphere.belongs(data))
fig = plt.figure(figsize=(10,10))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
geomstats.visualization.plot(data[15:20], ax=ax, space="S2", label=names[15:20], s=80, alpha=0.5)
fig.savefig('tbd00.png')


data, img_paths = geomstats.datasets.utils.load_poses()
so3 = geomstats.geometry.special_orthogonal.SpecialOrthogonal(n=3, point_type="vector", equip=False)
assert np.all(so3.beslongs(data)) #only shape is checked
# data (np,float64,(5,3))
img1 = matplotlib.image.imread(os.path.join(geomstats.datasets.utils.DATA_PATH, "poses", img_paths[0]))
img2 = matplotlib.image.imread(os.path.join(geomstats.datasets.utils.DATA_PATH, "poses", img_paths[1]))
fig = plt.figure(figsize=(16, 8))
fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(16,8))
ax0.imshow(img1)
ax1.imshow(img2)
ax0.axis("off")
ax1.axis("off")
fig.tight_layout()
fig.savefig('tbd00.png')


karate_graph = geomstats.datasets.utils.load_karate_graph()
hyperbolic_embedding = geomstats.datasets.prepare_graph_data.HyperbolicEmbedding(max_epochs=20)
embeddings = hyperbolic_embedding.embed(karate_graph)
disk = geomstats.visualization.PoincareDisk(coords_type="ball")
fig, ax = plt.subplots(figsize=(8, 8))
disk.set_ax(ax)
disk.draw(ax=ax)
ax.scatter(embeddings[:, 0], embeddings[:, 1])
fig.savefig('tbd00.png')


data, patient_ids, labels = geomstats.datasets.utils.load_connectomes()
# data (np,float64,(86,28,28))
assert np.abs(data-data.transpose(0,2,1)).max() < 1e-10
assert np.linalg.eigvalsh(data)[:,0].min() > 0 #all PD matrices (positive definite)
manifold_pd = geomstats.geometry.spd_matrices.SPDMatrices(data.shape[1], equip=False)
assert np.all(manifold_pd.belongs(data)) #True
labels_str = ["Healthy", "Schizophrenic"]
fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
ax0.imshow(data[0])
ax0.set_title(labels_str[labels[0]])
ax1.imshow(data[1])
ax1.set_title(labels_str[labels[1]])
fig.tight_layout()
fig.savefig('tbd00.png')

nerves, labels, monkeys = geomstats.datasets.utils.load_optical_nerves()
# nerves (np,float64,(22,5,3))
# labels (np,int64,(22,))
# monkeys (np,int64,(22,))
INDEX = 0
label_to_str = {0: "Normal nerve", 1: "Glaucoma nerve"}
label_to_color = {
    0: (102 / 255, 178 / 255, 255 / 255, 1.0),
    1: (255 / 255, 178 / 255, 102 / 255, 1.0),
}
fig = plt.figure(figsize=(8,8))
ax = mpl_toolkits.mplot3d.Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_xlim((2000, 4000))
ax.set_ylim((1000, 5000))
ax.set_zlim((-600, 200))
for nerve, label in zip(nerves[monkeys==INDEX], labels[monkeys==INDEX]):
    poly = mpl_toolkits.mplot3d.art3d.Poly3DCollection(nerve[np.newaxis], alpha=0.5)
    poly.set_color(matplotlib.colors.rgb2hex(label_to_color[int(label)]))
    poly.set_edgecolor("k")
    ax.add_collection3d(poly)
patch_0 = matplotlib.patches.Patch(color=label_to_color[0], label=label_to_str[0], alpha=0.5)
patch_1 = matplotlib.patches.Patch(color=label_to_color[1], label=label_to_str[1], alpha=0.5)
ax.legend(handles=[patch_0, patch_1], prop={"size": 20})
fig.savefig('tbd00.png')
