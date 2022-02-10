import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import sobel
from skimage.color import label2rgb

from skimage.morphology import dilation, closing, opening

coins = data.coins()
hist = np.histogram(coins, bins=np.arange(0, 256))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
axes[0].axis('off')
axes[1].plot(hist[1][:-1], hist[0], lw=2)
axes[1].set_title('histogram of grey values')

# thresholding
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

axes[0].imshow(coins > 100, cmap=plt.cm.gray, interpolation='nearest')
axes[0].set_title('coins > 100')
axes[0].axis('off')
axes[0].set_adjustable('box-forced')

axes[1].imshow(coins > 150, cmap=plt.cm.gray, interpolation='nearest')
axes[1].set_title('coins > 150')
axes[1].axis('off')
axes[1].set_adjustable('box-forced')
plt.tight_layout()

# edge-based segmentation
edges = canny(coins)
#edges = dilation(edges,np.ones((3,3)))
edges = closing(edges,np.ones((3,3)))
#edges = opening(edges,np.ones((3,3)))
fill_coins = ndi.binary_fill_holes(edges)
coins_cleaned = morphology.remove_small_objects(fill_coins, 21)

fig, axes = plt.subplots(1,3,figsize=(40, 10))
axes[0].imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
axes[0].set_title('Canny detector')
axes[0].axis('off')
axes[0].set_adjustable('box-forced')

axes[1].imshow(fill_coins, cmap=plt.cm.gray, interpolation='nearest')
axes[1].set_title('filling the holes')
axes[1].axis('off')

axes[2].imshow(coins_cleaned, cmap=plt.cm.gray, interpolation='nearest')
axes[2].set_title('removing small objects')
axes[2].axis('off')
axes[2].set_adjustable('box-forced')
plt.tight_layout()

# region-based segmentation
elevation_map = sobel(coins)

markers = np.zeros_like(coins)
markers[coins < 30] = 1
markers[coins > 150] = 2

segmentation = morphology.watershed(elevation_map, markers)

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_coins, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_coins, image=coins)

fig, axes = plt.subplots(2,3,figsize=(40, 20))
axes[0,0].imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
axes[0,0].set_title('elevation map')
axes[0,0].axis('off')
axes[0,0].set_adjustable('box-forced')

axes[0,1].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
axes[0,1].set_title('markers')
axes[0,1].axis('off')
axes[0,1].set_adjustable('box-forced')

axes[0,2].imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
axes[0,2].set_title('segmentation')
axes[0,2].axis('off')
axes[0,2].set_adjustable('box-forced')

axes[1,0].imshow(coins, cmap=plt.cm.gray, interpolation='nearest')
axes[1,0].contour(segmentation, [0.5], linewidths=1.2, colors='y')#warning
axes[1,0].axis('off')
axes[1,0].set_adjustable('box-forced')

axes[1,1].imshow(image_label_overlay, interpolation='nearest')
axes[1,1].axis('off')
axes[1,1].set_adjustable('box-forced')

plt.tight_layout()
