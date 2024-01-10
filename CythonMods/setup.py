from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#Cython conflicts with __init__.py files
extension = [Extension("NK_landscape", ["NK_landscape.pyx"]),
              Extension("graph", ["graph.pyx"])]

setup(
    ext_modules=cythonize(extension), 
    include_dirs=[numpy.get_include()] 
)