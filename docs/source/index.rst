NEDAS : Next-generation Ensemble Data Assimilation System

Introduction
============

Data assimilation (DA) combines information from model forecasts and observations
to obtain the best estimate of a dynamical system.
To improve the prediction skill of the Earth-system models, DA algorithms face two main challenges:
1) the **increasing size** of model state and observations demands computationally efficient algorithms; 
2) the **nonlinearity** in error growth mechanisms and in state-observation relation requires more sophisticated algorithms.

NEDAS provides a light-weight solution for developing new DA algorithms for Earth-system models.
To address the first challenge, **parallel computation** is implemented with
the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ package to ensure scalability to large-dimensional problems;
scientific computation packages, such as `numpy <https://numpy.org/>`_,
along with the `numba.jit compilation <https://numba.pydata.org/numba-doc/dev/reference/jit-compilation.html>`_ technology,
ensure computational efficiency.
As for the second challenge, NEDAS employs a **modular design** that separates the DA workflow into managable parts,
which can be upgraded with novel approaches to tackle with the nonlinear problems.
NEDAS now offers a collection of state-of-the-art DA algorithms, 
including the serial approaches that assimilate one observation at a time (similar to `DART <https://github.com/NCAR/DART>`_),
and the batch assimilation approaches (similar to the LETKF in `PDAF <https://pdaf.awi.de/trac/wiki>`_).
Thanks to the rich Python scientific computing ecosystem, NEDAS provides a flexible platform
for the rapid prototyping and testing of innovative DA methods, including the integratino of machine learning to enhance traditional algorithms.
Its intuitive user interface enables researchers to evaluate new DA concepts within real models at an early stage,
well before committing resources to full-scale operational implementation.
Moreover, NEDAS supports systematic intercomparison of DA methods, helping to gain theoretical insights
and guide the development of more robust and effective assimilation techniques.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   config_file
   examples

.. toctree::
   :maxdepth: 2
   :caption: User interfaces

   NEDAS.models
   NEDAS.dataset
   NEDAS.job_submitters

.. toctree::
   :maxdepth: 2
   :caption: API documentation

   NEDAS.schemes
   NEDAS.assim_tools
   NEDAS.config
   NEDAS.grid
   NEDAS.diag
   NEDAS.utils
