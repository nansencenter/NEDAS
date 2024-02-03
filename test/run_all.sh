#!/bin/bash

##test parallel lib with 4 processors
mpiexec -np 4 python -m unittest test_parallel.py

##test parallel lib with 1 processor
python -m unittest test_parallel.py



