NEDAS : Next-generation Ensemble Data Assimilation System

Introduction
============

Data assimilation is a 

NEDAS provides a light-weight Python solution to the ensemble data assimilation problem for geophysical models. 
It allows DA researchers to test and develop new DA ideas early-on in real models, 
before committing resources to full implementation in operational systems. 
NEDAS is armed with parallel computation (mpi4py) and pre-compiled numerical libraries (numpy, numba.njit) to ensure runtime efficiency. 
The modular design allows the user to add customized algorithmic components to enhance the DA performance. 
NEDAS offers a collection of state-of-the-art DA algorithms, including serial assimilation approaches (similar to DART), 
and batch assimilation approaches (similar to the LETKF in PDAF),
making it easy to benchmark new methods with the classic methods in the literature.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   config_file
   examples

.. toctree::
   :maxdepth: 2
   :caption: User interfaces

   models
   dataset
   job_submitters

.. toctree::
   :maxdepth: 2
   :caption: API documentation

   NEDAS.config
   NEDAS.grid
