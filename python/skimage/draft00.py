import numpy as np
import matplotlib.pyplot as plt

from skimage.filters import sobel
from skimage.measure import label
from skimage.segmentation import slic, join_segmentations
from skimage.morphology import watershed
from skimage.color import label2rgb
from skimage import data

coins = data.coins()
edges = sobel(coins)# edge-detection and watershed

markers = np.zeros_like(coins)
foreground, background = 1, 2
markers[coins < 30.0] = 2#background
markers[coins > 150.0] = 1#foreground

seg1 = label(watershed(edges, markers) == 1)
# SLIC superpixels segmentation
seg2 = slic(coins, n_segments=117, max_iter=160, sigma=1, compactness=0.75, multichannel=False)
segj = join_segmentations(seg1, seg2)

hFig, tmp1 = plt.subplots(2,2,True,True,figsize=(15,9),subplot_kw={'adjustable': 'box-forced'})

ax = tmp1.ravel()
ax[0].imshow(coins, cmap='gray')
ax[0].set_title('Image')

ax[1].imshow(label2rgb(seg1, image=coins, bg_label=0))
ax[1].set_title('Sobel+Watershed')

ax[2].imshow(label2rgb(seg2, image=coins, image_alpha=0.5))
ax[2].set_title('SLIC superpixels')

ax[3].imshow(label2rgb(segj, image=coins, image_alpha=0.5))
ax[3].set_title('Join')

for a in ax:
    a.axis('off')
hFig.tight_layout()
plt.show()
