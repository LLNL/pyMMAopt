#!/usr/bin/env python

import os
from setuptools import setup, find_packages

base_dir = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(base_dir, "pyMMAopt", "__about__.py"), "rb") as f:
    exec(f.read(), about)

if __name__ == "__main__":
    setup(
    name="pyMMAopt",
    version=about["__version__"],
    packages=find_packages(),
    author=about["__author__"],
    author_email=about["__email__"],
    install_requires=["numpy", "numexpr"],
    description="MMA optimization algorithm in python",
    license=about["__license__"],
    classifiers=[
        about["__license__"],
        about["__status__"],
        # See <https://pypi.org/classifiers/> for all classifiers.
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3")
