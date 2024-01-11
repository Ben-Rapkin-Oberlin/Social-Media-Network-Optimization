#!/bin/bash
cd CythonMods
python setup.py build_ext --inplace
cd ..
python run.py
