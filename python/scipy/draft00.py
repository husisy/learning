'''reference: https://docs.scipy.org/doc/numpy/user/quickstart.html'''

import numpy as np

np.show_config()

## property
z0 = np.random.randn(2, 3)
z0.ndim
z0.shape
z0.size
z0.itemsize #in byte
z0.dtype #np.int32 np.float64 np.complex128
z0.dtype.itemsize #the same as above
z0.dtype.name
z0.flat
z0.T
# z0.data
## method
z0.reshape((3,2)) #return a view
# z0.resize((3,2)) #in-place
z0.ravel()
z0.transpose(0,1)

## method
z0.sum
z0.min
z0.max
z0.cumsum


## create
np.array([1,2,3], dtype=np.float64) #np.asarray
np.zeros((2,3)) #np.zeros_like
np.ones((2,3)) #np.ones_like
np.empty((2,3)) #uninitialized random #np.empty_like
np.arange(3)
np.linspace(0, 1, 100) #np.logspace
np.fromfunction
np.eye
np.mgrid #np.ogrid


## print
# np.set_printoptions(precision=2, linewidth=150)


## basic operation
# + - * / @ **
# += *=
# TODO upcasting #np.promote_types np.result_type


## logical
# < > <= >=
np.all
np.any
np.nonzero
np.where


## ordering
np.argmax
np.argmin
np.max
np.min
np.median
np.maximum
np.minimum
np.argsort
np.sort
np.ptp
np.searchsorted


## cast
np.astype
np.ceil
np.floor
np.rint
np.trunc
np.around


## universal function
np.exp
np.sqrt
np.sin
np.add
# apply_along_axis average bincount clip conj corrcoef cov cross cumprod
# cumsum diff inner inv lexsort mean outer prod re std sum trace var vdot vectorize


## special constant
np.pi
np.newaxis


## stack
np.concatenate
np.stack
# np.hstack np.vstack np.column_stack np.row_stack np.r_ np.c_


## shape manipunation
# .ravel
# .flatten
# .reshape
# .resize
# .transpose
# .T
# .repeat


## split
np.array_split #np.vsplit np.hsplit np.split


## view (shallow copy)
z0 = np.random.rand(2, 3)
z0_view = z0.view()
z0_view.base is z0 #True
z0_view.flags.owndata
z0_view = z0.reshape((3,2))
z0_view = z0[1:]
# deep copy
z0_copy = z0.copy()
# sometimes `.copy()` should be called if the original array is not requred anymore
#   see https://numpy.org/devdocs/user/quickstart.html#deep-copy


## dtype
np.integer
np.floating
np.bool_
np.byte
np.ubyte
np.short
np.ushort
np.intc
np.int_
np.uint
np.longlong
np.ulonglong
np.half
np.float16
np.float_
np.single
np.double
np.longdouble
np.csingle
np.cdouble
np.clongdouble

np.iinfo(int)
np.iinfo(np.int32)
np.finfo(float)
np.finfo(np.float32)
