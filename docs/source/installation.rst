Installation
============

Dependencies
------------

NEDAS core requires Python >=3.10, the following packages are mandatory:

- `numpy <https://numpy.org>`_
- `scipy <https://scipy.org>`_
- `matplotlib <https://matplotlib.org/>`_
- `pyproj <https://pyproj4.github.io/pyproj/stable/>`_
- `pyshp <https://github.com/GeospatialPython/pyshp>`_
- `netCDF4 <https://unidata.github.io/netcdf4-python/>`_
- `pyYAML <https://pyyaml.org/>`_
- `xarray <https://docs.xarray.dev/en/stable/>`_
- `pandas <https://pandas.pydata.org/>`_

The dynamical model, unless directly implemented in Python, needs to be installed separately.
Check its own documentation for details on installation.
Note that for individual model and dataset module implementation, a higher Python version and additional libraries may be required.

Optional Features
-----------------

To enable MPI support for parallel processing, make sure to install the MPI library
(e.g. MPICH or intel OpenMP) and install the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ package.
If mpi4py is not available, NEDAS will automatically fall back to serial processing mode.

If `numba <https://numba.pydata.org/>`_ is installed,
some core algorithms within NEDAS will be JIT-compiled to machine code at runtime to improve efficiency.

An alternative FFT implementation is enabled by `pyFFTW <https://pyfftw.readthedocs.io/en/latest/>`_.
If pyFFTW is not available, NEDAS will fall back to the numpy.fft package.

The ``DART`` assimilator calls the analysis kernels of
`DART <https://github.com/NCAR/DART>`_ (NCAR's Data Assimilation Research Testbed) directly,
instead of NEDAS's own implementation of the same algorithm.
DART is not a Python package and cannot be installed with pip; it is a Fortran code base from
which a shared library is built, see `Build the DART kernels`_ below.
It is only needed if ``assimilator_def.type`` is set to ``DART``.

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

Build the DART kernels
----------------------

The ``DART`` assimilator runs the analysis kernels compiled from
`DART <https://dart.ucar.edu/>`_ instead of NEDAS's native implementation, so that
NEDAS's filters stay numerically consistent with the upstream ones.
It is only needed if you set ``assimilator_def.type`` to ``DART``.

DART is not distributed on PyPI or conda, and has to be obtained and compiled separately.
Follow the `DART documentation <https://docs.dart.ucar.edu/>`_ to download it and to
configure ``build_templates/mkmf.template`` for your compiler and netCDF installation
(this is the usual DART setup step; NEDAS cannot guess these settings).
A Fortran compiler and netcdf-fortran are required.

.. tip::

   The simplest way to avoid the linking pitfalls below is to build DART against a
   **serial** netCDF installed in the same environment that runs NEDAS -- for example
   ``conda install -c conda-forge netcdf-fortran gfortran`` into a dedicated environment,
   with ``python_env`` pointing at it. A cluster's module-built netCDF is usually compiled
   against MPI, which drags in that MPI (and, if it is GPU-aware, the CUDA runtime) as
   transitive dependencies of the kernel library, none of which DART needs here: the
   library is built serial (``null_mpi``) and never calls MPI. A serial netCDF has no such
   chain, and its libraries are already on the path at runtime. Note that conda's
   ``netcdf.mod`` must be read by the same compiler that produced it, so build DART with
   that environment's ``gfortran``.

.. important::

   This applies when you build against a module-provided netCDF. Add
   ``-Wl,-rpath,<netcdf>/lib`` to the ``LIBS`` line of your ``mkmf.template``, alongside
   the usual ``-L<netcdf>/lib``. NEDAS loads the resulting library with ``ctypes`` from inside
   a Python (often conda) environment, whose ``LD_LIBRARY_PATH`` typically does not carry the
   netCDF paths that were present at link time. Without the rpath the build succeeds but the
   library fails to load with ``OSError: libnetcdff.so...: cannot open shared object file``.
   On a cluster where netCDF comes from modules, remember that the Fortran and C netCDF
   libraries may live in separate prefixes, each needing its own ``-L`` and ``-Wl,-rpath``.

   An rpath fixes the libraries NEDAS's own library links against directly, but it cannot
   fix all of them. A module-built netCDF pulls in HDF5, MPI, compression and sometimes the
   CUDA runtime, and when one of those intermediate libraries carries ``RUNPATH`` (as
   module-built MPI typically does) the loader resolves *its* dependencies using only its own
   ``RUNPATH`` and ``LD_LIBRARY_PATH`` -- our rpath is ignored there. Those libraries therefore
   have to be visible at runtime: load the same modules that DART was built with, or add their
   lib directories to ``LD_LIBRARY_PATH``, in the environment that runs NEDAS.

   Run ``ldd libdartkernels.so`` in the *same* environment that will import NEDAS (with the
   conda environment activated, if you use one) to confirm nothing is left unresolved. This is
   the quickest check, since the build itself succeeds either way.

Once DART is in place, build the shared library that NEDAS loads:

.. code-block:: bash

   export NETCDF=/path/to/netcdf     # netcdf-fortran prefix
   cd NEDAS/assim_tools/assimilators/DART
   ./build_dart_kernels.sh --dart /path/to/DART

This reuses DART's own build machinery and produces ``libdartkernels.so`` next to the script,
where the assimilator looks for it by default.
To keep it elsewhere, pass ``-o /path/to/libdartkernels.so`` and point
``assimilator_def.dart_lib`` at that path.

.. note::

   The build adds a single ``public ::`` line to DART's ``assim_tools_mod.f90``, since the
   kernels NEDAS calls are private to that module upstream. The edit is idempotent and marked
   with a comment; running ``git checkout`` in your DART directory reverts it.

.. warning::

   DART reports fatal conditions by calling its own ``error_handler``, which terminates the
   process. Reached from inside a kernel, that ends the NEDAS run immediately -- no python
   exception, no traceback, just DART's message and a stop. The wrapper guards the cases it
   can check up front (a degenerate observation, an unknown ``filter_kind``) and returns a
   status instead, but it cannot intercept every check inside DART. Treat an abrupt stop
   with a DART message in the log as an input the kernel rejected, not as a NEDAS crash.

Select the kernel with ``assimilator_def.filter_kind``: ``EAKF`` (default), ``PARTICLE``,
``RHF``, ``GAMMA`` or ``BNRHF``. ``BNRHF`` is the bounded kernel and also reads
``bounded_below``/``bounded_above`` and ``lower_bound``/``upper_bound`` from the same
section.

``ENKF``, ``KERNEL`` and ``KDE`` exist in DART but are refused for now with a
``NotImplementedError``. All three need DART's utilities subsystem initialized: the first
two build their random seed from ``my_task_id()``, which initializes the utilities and reads
``input.nml`` on the way, while ``KDE`` reads its own ``kde_nml`` and refuses to proceed
until the utilities are up. This interface does not initialize DART, which is what keeps the
kernels free of runtime setup and lets ``EAKF`` reproduce NEDAS's native results exactly.

The kernels themselves are fine: all three run once ``initialize_utilities()`` has been
called and an ``input.nml`` exists in the working directory. Enabling them means adding an
initialization entry point, putting an ``input.nml`` in every rank's working directory (the
filename is hardcoded and looked up relative to the current directory), and accepting the
``dart_log.out``/``dart_log.nml`` that DART writes there. Until then the guard matters,
because otherwise DART calls its ``error_handler`` and stops the run outright.

If the libraries DART needs are awkward to provide in your usual NEDAS environment, keep a
separate environment holding them and point :ref:`python_env <config_file>` at its source
script; NEDAS sources that file when launching its job steps.

``tests/test_dart_kernels.py`` checks the DART kernels against NEDAS's native EAKF
implementation, and is skipped unless the library has been built. Only ``EAKF`` has a
native NEDAS counterpart to compare against -- the other kernels are covered by lighter
smoke tests, so they would not catch a subtle numerical change upstream.

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
