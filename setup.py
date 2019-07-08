#!/usr/bin/env python

import os
import pkg_resources
import sys

from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as f:
    readme = f.read()

install_requires = [
    'cached_property',
    'torch>=1.0.0',
    'joblib>=0.11',
    'gym>=0.10.5',
    'numpy>=1.13.3',
    'terminaltables',
    'pandas',
    'pybullet==2.4.1',
    'redis',
]

setup(
    name='meta_rl_tools',
    version='0.0.1',
    long_description=readme,
    author='Hoge',
    author_email='hoge@fuga.co.jp',
    url='https://github.com/rlsim2realarchitecture/rlsim2realarchitecture',
    packages=find_packages(),
    zip_safe=False,
    install_requires=install_requires,
)
