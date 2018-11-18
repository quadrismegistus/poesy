import os,sys
from setuptools import setup
_here = os.path.abspath(os.path.dirname(__file__))

with open("README.rst", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [x.strip() for x in fh.read().split('\n') if x.strip()]

setup(
    name='poetix',
    version='0.0.9',
    description=('Poetix: Poetic processing, for Python'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ryan Heuser',
    author_email='heuser@stanford.edu',
    url='https://github.com/quadrismegistus/poetix',
    license='MPL-2.0',
    packages=['poetix','prosodic'],
    install_requires=requirements,
    include_package_data=True,
    classifiers=[
        #'Development Status :: 3 - Alpha',
        #'Intended Audience :: Science/Research',
        #'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3.6'
    ],
    )
