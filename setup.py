#!\usr\bin\python
"""Setuptools-based installation."""

from codecs import open  # To use a consistent encoding
from pathlib import Path

from setuptools import find_packages, setup


try:
    from m5 import __version__
except Exception:
    with (Path(__file__).parent / "m5").open() as f:
        __version__ = f.read()


description = "M5 time series modelling competition."
try:
    here = Path(__file__).absolute()
    with open(here / "README.md", encoding="utf-8") as f:
        long_description = f.read()
except Exception:
    long_description = description


#

setup(
    name="m5",
    packages=find_packages(),
    description=description,
    long_description=long_description,
    author="Anatoly Makarevich and Yuliya Malakhouskaya",
    version=__version__,
    package_data={"": ["m5/VERSION"]},
    include_package_data=True,
    # NOTE: Actual requirements in environment.yml
    install_requires=[],
    entry_points=dict(),
)
