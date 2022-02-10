import setuptools
import Cython.Build

tmp0 = [
    setuptools.Extension('myqueue', sources=['python/queue.pyx']),
]

setuptools.setup(
    name='my_cython_pkg',
    ext_modules = Cython.Build.cythonize(tmp0),
    # zip_safe=False,
)
