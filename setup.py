#!/usr/bin/env python

from setuptools import setup, find_packages

# read the contents of your README file
from os import path
from os import path as OSPath

VERSION = '1.0.0'

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = open(path.join(this_directory, 'requirements.txt'), encoding='utf-8').read().strip().split('\n')

setup(name='PVcharacterization_Utils',
      version='1.0.0',
      description='toolbox to process pv flashtest data and pv electroluminescence images',
      long_description=long_description,
      long_description_content_type='text/markdown',
      include_package_data = True,
      license = 'MIT',
      classifiers = [
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Physics',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research'
        ],
      keywords = 'Image, data processing, solar cell, flash test, electroluminescence',
      install_requires = install_requires,
      author= 'PV_team',
      author_email= 'francois.bertin7@wanadoo.fr',
      url= 'https://github.com/Bertin-fap/PVcharacterization',
      packages=find_packages(),
      )
