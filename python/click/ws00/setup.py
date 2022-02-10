from setuptools import setup

setup(
    name='click-example-bashcompletion',
    version='1.0',
    py_modules=['bashcompletion'],
    install_requires=[
        'click',
    ],
    entry_points={
        'console_scripts': ['bashcompletion=bashcompletion:cli'],
    },
)
