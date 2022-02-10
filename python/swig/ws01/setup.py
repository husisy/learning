# -*- coding:utf-8 -*-
import os
import setuptools

hf_file = lambda *x: os.path.join('python','my_swig_pkg00',*x)

swig_c_ext = setuptools.Extension('my_swig_pkg00._example', [hf_file('example.c'), hf_file('example.i')])
swig_cpp_ext = setuptools.Extension(
    'my_swig_pkg00._stl_example',
    sources=[hf_file('stl_example.cpp'), hf_file('stl_example.i')],
    swig_opts=['-c++'],
    extra_compile_args=['-std=c++11']
)
setuptools.setup(
    name='my_swig_pkg00',
    version='0.1.0',
    packages=setuptools.find_packages('python'),
    package_dir={'': 'python'},
    ext_modules=[swig_c_ext, swig_cpp_ext],
)
