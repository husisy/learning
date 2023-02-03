import os
import json
import setuptools

with open(os.path.join('python','mypackage','_package.json'), 'r', encoding='utf-8') as fid:
    _package = json.load(fid)
__version__ = _package['version']

with open('README.md', encoding='utf-8') as fid:
    long_description = fid.read()

setuptools.setup(
    name='mypackage', #this is NOT python-module name, one package could include mutiple modules
    version=__version__,
    package_dir={'':'python'},
    packages=setuptools.find_packages('python'),
    # https://setuptools.pypa.io/en/latest/userguide/datafiles.html
    package_data={"mypackage": ["*.json"]},
    scripts=['scripts/myscripts'],
    description='an example package',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='test00',
    author_email='test00@test.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # install_requires=['numpy>=1.0'],
)
