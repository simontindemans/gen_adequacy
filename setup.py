from setuptools import setup

# Based on tutorial https://packaging.python.org/tutorials/packaging-projects/

# Get the long description from the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gen_adequacy',
    version='0.1.0',
    packages=['gen_adequacy'],
    url='https://github.com/simontindemans/gen_adequacy',
    author='Simon Tindemans',
    author_email='s.h.tindemans@tudelft.nl',
    description='Tools for single node generation adequacy analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],
    install_requires=['numpy', 'numba'],
    python_requires='>=3.5',
)
