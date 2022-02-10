import setuptools
import Cython.Build


# setuptools.Extension(xxx, libraries=["m"]) #Unix-like specific, libc.math
tmp0 = [
    setuptools.Extension('my_cython_pkg', sources=['python/my_cython_pkg.pyx']),
    setuptools.Extension('my_cython_cpp_pkg', sources=['python/my_cython_cpp_pkg.pyx']),
]

setuptools.setup(
    name='my_cython_pkg',
    ext_modules=Cython.Build.cythonize(tmp0, annotate=True),
    zip_safe=False,
)
