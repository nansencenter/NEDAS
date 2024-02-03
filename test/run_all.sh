#!/bin/bash
set -a; . ../config/defaults
echo
echo 'test parallel lib:'
echo 'test with 4 processors'
mpiexec -np 4 python -m unittest test_parallel.py
echo
echo 'test with 1 processor'
python -m unittest test_parallel.py
echo
echo 'test Grid class:'
python -m unittest test_grid.py
echo
echo 'test analysis lib:'
python -m unittest test_analysis.py
echo
