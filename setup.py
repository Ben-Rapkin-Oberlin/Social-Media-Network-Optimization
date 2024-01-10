from setuptools import setup
from Cython.Build import cythonize
import numpy
setup(
    ext_modules=cythonize(["NK_landscape.pyx", "graph.pyx"]), 
    include_dirs=[numpy.get_include()] 
)