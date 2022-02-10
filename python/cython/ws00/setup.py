import setuptools
import Cython.Build

tmp0 = setuptools.Extension('my_cython_pkg',
        sources=['python/my_cython_pkg.pyx'],
)
setuptools.setup(
    name='my_cython_pkg',
    ext_modules=Cython.Build.cythonize([tmp0]),
    zip_safe=False,
)
