import sys
import setuptools
import pybind11
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

__version__ = '0.0.1'
assert sys.platform=='linux', 'win-link should be test with cmake'

ROOT_DIR = os.path.dirname(__file__)
hf_file = lambda *x,_dir=ROOT_DIR: os.path.join(_dir, *x)

class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    def build_extensions(self):
        assert self.compiler.compiler_type=='unix', 'win-link should be test with cmake'
        opts = ['-DVERSION_INFO="{}"'.format(self.distribution.get_version()), '-fvisibility=hidden']
        link_opts = ['-Llib', '-lutils']
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)

ext_modules = [
    Extension(
        'zctest_link._cpp',
        ['src/main.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            hf_file('include'),
        ],
        language='c++'
    ),
]

setup(
    name='zctest_link',
    version=__version__,
    description='A test project using pybind11',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.4'],
    setup_requires=['pybind11>=2.4'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
