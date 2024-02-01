from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#Cython conflicts with __init__.py files
extension = [Extension("new_graph", ["new_graph.pyx"])]

setup(
    ext_modules=cythonize(extension), 
    include_dirs=[numpy.get_include()] 
)

#to run:
#python setup.py build_ext --inplace
