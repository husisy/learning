from skimage import data, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt


img = data.coffee()

labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1)
labels2 = graph.cut_threshold(labels1, g, 29)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(2,1,True,True,figsize=(8, 12))
ax[0].imshow(out1)
ax[1].imshow(out2)
for a in ax:
    a.axis('off')
plt.tight_layout()
