# skimage

1. link
   * [official site](http://scikit-image.org/)
   * [user guide](http://scikit-image.org/docs/stable/user_guide.html)
   * [API reference](httxp://scikit-image.org/docs/stable/api/api.html)
   * [official examples](http://scikit-image.org/docs/stable/auto_examples/index.html#examples-gallery)
   * [HW1.ipynb](https://github.com/weiuniverse/cv_homework/blob/master/HW1Filters/src/HW1.ipynb)
2. plane-wise processing `for pln, image in enumerate(im3d)`
3. **never use `astype` on an image, instead**
   * `skimage.img_as_float()`
   * `skimage.img_as_ubyte()`
   * `skimage.img_as_uint()`
   * `skimage.img_as_int()`
   * `preserve_range` parameter
   * warning when loss precision (ignored by using context manager)
4. `OpenCV`: BGR, `uint8`
5. array order
   * prefetching: modern processors never retrieve just one item from memory, but rather a whole chunk of adjacent items
6. coordinate conventions
   * time
   * plane
   * row
   * column
   * channel

**Data type**|**Range**
:-----:|:-----:
uint8 | 0 to 255
uint16 | 0 to 65535
uint32 | 0 to $2^{32}$
float | -1 to 1 or 0 to 1
int8 | -128 to 127
int16 | -32768 to 32767
int32 | -$2^{31}$ to $2^{31}-1$

**Image type**|**coordinates**
:-----:|:-----:
2D grayscale|(row, col)
2D multichannel (eg. RGB)|(row, col, ch)
3D grayscale|(pln, row, col)
3D multichannel|(pln, row, col, ch)
2D color video | (t, row, col, ch)
3D multichannel video | (t, pln, row, col, ch)
