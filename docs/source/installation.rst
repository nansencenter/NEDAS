Installation
============

.. contents::
   :local:
   :depth: 2

Dependencies
------------

NEDAS requires Python >=3.8, the following packages are mandatory:

- `numpy <https://numpy.org>`_
- `scipy <https://scipy.org>`_
- `matplotlib <https://matplotlib.org/>`_
- `pyproj <https://pyproj4.github.io/pyproj/stable/>`_
- `pyshp <https://github.com/GeospatialPython/pyshp>`_
- `netCDF4 <https://unidata.github.io/netcdf4-python/>`_
- `pyYAML <https://pyyaml.org/>`_

The dynamical model, unless directly implemented in Python, needs to be installed separately.
Check its own documentation for details on installation.

Optional Features
-----------------

To enable MPI support for parallel processing, make sure to install the MPI library
(e.g. MPICH or intel OpenMP) and install the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ package.
If mpi4py is not available, NEDAS will automatically fall back to serial processing mode.

If `numba <https://numba.pydata.org/>`_ is installed,
some core algorithms within NEDAS will be JIT-compiled to machine code at runtime to improve efficiency.

An alternative FFT implementation is enabled by `pyFFTW <https://pyfftw.readthedocs.io/en/latest/>`_.
If pyFFTW is not available, NEDAS will fall back to the numpy.fft package.

Some NEDAS submodules may also require additional packages to be installed.
See the submodule documentation for more details.

Install via pip
---------------

NEDAS is available from the PyPI platform. To install the latested version:

.. code-block:: bash

   pip install NEDAS

You can install NEDAS with optional features by using **extras** in your pip command.
For example, to install **all** the optional dependencies:

.. code-block:: bash

   pip install NEDAS[all]

To install dependencies related to specific features, use one or more ``tag`` listed in the table below:

.. code-block:: bash

   # replace tag with one listed in the table below
   pip install NEDAS[tag]

+---------------+---------------------+-------------------------------------------------+
| Tag           | Additional packages | Purpose                                         |
+===============+=====================+=================================================+
| ``mpi``       | mpi4py              | MPI-based parallel processing                   |
+---------------+---------------------+-------------------------------------------------+
| ``jit``       | numba               | JIT compilation                                 |
+---------------+---------------------+-------------------------------------------------+
| ``fftw``      | pyFFTW              | Alternative implementation of FFT               |
+---------------+---------------------+-------------------------------------------------+
| ``grib``      | pygrib              | Support for GRIB data format                    |
+---------------+---------------------+-------------------------------------------------+
| ``alignment`` | opencv-python       | Optical flow algorithms in alignment technique  |
+---------------+---------------------+-------------------------------------------------+
| ``emulator``  | tensorflow, torch   | Machine learning algorithms for model emulators |
+---------------+---------------------+-------------------------------------------------+

Install via Conda
-----------------

If you prefer using Conda, we provide an ``environment.yml`` file to help you set up everything in a controlled environment:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate nedas

The ``environment.yml`` file only contains the miminal dependencies,
you can modify the file to include additional features, such as mpi4py, numba, etc.
You can also install them via pip in the conda environment afterwards.

Manual installation
-------------------

You can also download NEDAS from the Github repository and install it manually,
especially if you plan to contribute or develop your own features.

To do so, first you can fork the `NEDAS repository <https://github.com/nansencenter/NEDAS>`_ on GitHub to your own account,
then clone your fork and create a new development branch:

.. code-block:: bash

   # clone your fork (replace USERNAME with your GitHub username)
   git clone https://github.com/USERNAME/NEDAS.git`
   cd NEDAS

   # create and switch to a new development branch called 'my-feature'
   git checkout -b my-feature

You can install the NEDAS package in editable mode for development.

.. code-block:: bash

   pip install -e .

Or just specify the ``PYTHONPATH`` without even installing
(of course you need to install the dependencies in ``requirements.txt``).

.. code-block:: bash

   # add NEDAS package to python search path
   # replace INSTALL_PATH to the directory containing the cloned NEDAS package
   export PYTHONPATH=$PYTHONPATH:INSTALL_PATH/NEDAS

Now any changes you make in the code will immediately reflect in your Python environment.

Run NEDAS
---------

Once installed, the NEDAS analysis scheme can be run as:

.. code-block:: bash

   python -m NEDAS -c CONFIG_FILE.yml

``CONFIG_FILE.yml`` is the YAML configuration file, see :doc:`config_file` for more details.

Run in Docker containers
------------------------

If you don't want to deal with installation and just want to see NEDAS in action,
several examples come with Docker images that you can run immediately
if `docker <https://www.docker.com/>`_ is available on your machine.

For example, the :doc:`examples.qg` case provides a
`Docker image <https://hub.docker.com/r/myying/nedas-qgmodel-benchmark>`_.
You can directly pull it from DockerHub and give it a try.
