import matplotlib.pyplot as plt
from skimage import data, color, img_as_float
import numpy as np
from skimage.filters import rank

grayscale_image = img_as_float(data.camera())
image = color.gray2rgb(grayscale_image)

fig, (ax1, ax2) = plt.subplots(1,2,True,True,figsize=(16,8))
ax1.imshow([1,0,0]*image)
ax2.imshow([1,1,0]*image)
ax1.set_adjustable('box-forced')
ax2.set_adjustable('box-forced')


hue_gradient = np.linspace(0,1,50)
hsv = np.ones((1,50,3), dtype=float)
hsv[:,:,0] = hue_gradient
all_hues = color.hsv2rgb(hsv)
fig, ax = plt.subplots(figsize=(15,6))
ax.imshow(all_hues, extent=(0, 1, 0, 0.2))
ax.set_axis_off()

def colorize(image, hue, saturation=1):
    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)

image = color.gray2rgb(data.camera())
hue_rotations = np.linspace(0, 1, 6)
fig, axes = plt.subplots(2,3,True,True)
for ax, hue in zip(axes.flat, hue_rotations):
    tinted_image = colorize(image,hue,0.3)
    ax.set_axis_off()
    ax.set_adjustable('box-forced')
fig.tight_layout()


top_left = (slice(100),) * 2
bottom_right = (slice(-100, None),) * 2

sliced_image = image.copy()
sliced_image[top_left] = colorize(image[top_left], 0.82, saturation=0.5)
sliced_image[bottom_right] = colorize(image[bottom_right], 0.5, saturation=0.5)

# Create a mask selecting regions with interesting texture.
noisy = rank.entropy(grayscale_image, np.ones((9, 9)))
textured_regions = noisy > 4
# Note that using `colorize` here is a bit more difficult, since `rgb2hsv`
# expects an RGB image (height x width x channel), but fancy-indexing returns
# a set of RGB pixels (# pixels x channel).
masked_image = image.copy()
masked_image[textured_regions, :] *= [1,0,0]

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True, sharey=True)
ax1.imshow(sliced_image)
ax2.imshow(masked_image)
ax1.set_adjustable('box-forced')
ax2.set_adjustable('box-forced')
