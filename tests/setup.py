#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='metalprot',
    version='0.0.0',
    author='Lei Lu',
    author_email='lonelur@gmail.com',
    url='https://github.com/lonelu//Metalprot',
    description='metal binding protein design',
    license='MIT',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'numpy',
        'matplotlib',
        'prody',
        'scikit-learn', # edited by 5.8
        'numba',
        'scipy',
        #'dataclasses',
    ],
    # extras_require = {
    #     '':  [],
    # },
    entry_points={
        'console_scripts': [ ],
    },
    
    #long_description=open('README.md').read(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
    ],
)
