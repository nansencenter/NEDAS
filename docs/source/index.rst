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
which can be upgraded with new approaches to tackle with the nonlinear problems.
Thanks to the rich Python ecosystem for scientific computing and machine learning, NEDAS provides a flexible platform
for rapid prototyping of innovative DA methods.

NEDAS now offers a collection of DA algorithms for benchmarking and intercomparison,
including the serial approaches that assimilate one observation at a time (similar to `DART <https://github.com/NCAR/DART>`_),
and the batch assimilation approaches (similar to the LETKF in `PDAF <https://pdaf.awi.de/trac/wiki>`_).
Its intuitive user interfaces and interoperability with other DA software enable early testing of new DA algorithms,
well before committing resources to full-scale operational implementation.

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
