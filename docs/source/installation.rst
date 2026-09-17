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

The ``PDAF`` assimilator likewise runs the analysis kernels of
`PDAF <https://pdaf.awi.de>`_ (the Parallel Data Assimilation Framework), through its
`pyPDAF <https://github.com/yumengch/pyPDAF>`_ bindings.
pyPDAF is not on PyPI or conda-forge and is built from source against a PDAF release,
see `Install pyPDAF`_ below.
It is only needed if ``assimilator_def.type`` is set to ``PDAF``.

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

Select the kernel with ``assimilator_def.filter_kind``: ``EAKF`` (default), ``ENKF``,
``KERNEL``, ``PARTICLE``, ``RHF``, ``GAMMA``, ``BNRHF`` or ``KDE``. The bounded kernels
(``BNRHF``, ``KDE``) also read ``bounded_below``/``bounded_above`` and
``lower_bound``/``upper_bound`` from the same section.

The stochastic kernels (``ENKF``, ``KERNEL``) draw from a random sequence held inside the
DART library, which ``numpy``'s generator cannot reach. NEDAS seeds it through
``assimilator_def.random_seed``: ``0`` derives a seed from the analysis time, which is
identical on every rank -- necessary, because every rank computes the increment for the same
observation, so a rank-dependent stream would perturb one observation differently in
different parts of the domain -- while changing from cycle to cycle, so the same draws are
not replayed. Set a non-zero value to pin a run instead.

Some kernel options live in DART's namelists rather than in its kernel arguments. These are
exposed as ``assimilator_def`` entries and written into an ``input.nml`` for DART to read:
``sort_obs_inc`` (``ENKF``), ``rectangular_quadrature`` and ``gaussian_likelihood_tails``
(``RHF``), and ``quadrature_order`` (``KDE``).

Only those three kernels trigger this, since only they consult a namelist. The rest never
initialize DART at all, which keeps them free of runtime setup and lets ``EAKF`` reproduce
NEDAS's native results exactly. When it is triggered, NEDAS writes an ``input.nml`` into the
working directory (the filename is hardcoded and looked up relative to the current
directory) and DART writes ``dart_log.out`` and ``dart_log.nml`` beside it. An ``input.nml``
that NEDAS did not write is never overwritten -- remove it to let NEDAS manage the namelist,
or set ``write_input_nml: False`` and supply your own. A supplied file must contain
``&utilities_nml``, ``&assim_tools_nml`` and ``&obs_kind_nml``; the last is needed because
initializing the assimilation tools reaches DART's observation-kind module. Empty sections
are fine -- they simply leave DART's defaults in place.

.. note::

   DART reports a fatal condition by calling its ``error_handler``, which stops the process:
   there is no exception for Python to catch, and a NEDAS run would end with only DART's
   message in the log. NEDAS therefore checks what it can in advance (a missing file, a
   missing section, an unsupported option) and rehearses the initialization itself in a
   subprocess, so a namelist DART dislikes surfaces as an ordinary Python exception carrying
   DART's own message. The per-observation kernel calls are too frequent to guard that way
   and instead return status codes, which NEDAS raises as exceptions.

``sampling_error_correction`` is not supported yet. It applies to every kernel, since it
changes the regression coefficient in ``update_from_obs_inc``, but beyond the namelist flag
DART also needs its correction table (``sampling_error_correction_table.nc``) staged at
runtime. Setting it raises rather than silently regressing with an unpopulated table.

If the libraries DART needs are awkward to provide in your usual NEDAS environment, keep a
separate environment holding them and point :ref:`python_env <config_file>` at its source
script; NEDAS sources that file when launching its job steps.

``tests/test_dart_kernels.py`` checks the DART kernels against NEDAS's native EAKF
implementation, and is skipped unless the library has been built. Only ``EAKF`` has a
native NEDAS counterpart to compare against -- the other kernels are covered by lighter
smoke tests, so they would not catch a subtle numerical change upstream.

Install pyPDAF
--------------

The ``PDAF`` assimilator hands each analysis partition to `PDAF <https://pdaf.awi.de>`_'s
own domain-localized filters (LESTKF, LETKF, LSEIK, LNETF, LKNETF) instead of NEDAS's
native ETKF, through the `pyPDAF <https://github.com/yumengch/pyPDAF>`_ bindings.
It is only needed if you set ``assimilator_def.type`` to ``PDAF``.

pyPDAF has no PyPI or conda-forge package, so it is built from source against a PDAF
release; the build is meson-python over PDAF's Fortran library and needs a Fortran
compiler, MPI and BLAS/LAPACK:

.. code-block:: bash

   git clone --recursive https://github.com/yumengch/pyPDAF.git   # PDAF is a submodule
   cd pyPDAF
   FC=<mpi fortran wrapper> CC=<mpi c wrapper> pip install . --no-build-isolation \
       -Csetup-args="-Dincdirs=<blas include>" \
       -Csetup-args="-Dlibdirs=<blas lib>" \
       -Csetup-args="-Dblas_lib=openblas"

Build it inside the environment that runs NEDAS, with the same MPI that mpi4py was built
against -- a mismatch there surfaces as a hang or a crash at the first PDAF call rather
than as an import error. Note that meson does not search for MPI itself on Linux: MPI
comes entirely from the wrapper compilers named in ``FC``/``CC``.
``NEDAS/assim_tools/assimilators/PDAF/install_pypdaf_betzy.sh`` is a worked example of the
above (and of the module pitfalls behind it) for betzy; see ``install_pypdaf.md`` next to it.

Unlike the DART kernels, which are built serial, pyPDAF requires MPI: it is built against an
MPI library and initializes MPI on import, so the ``PDAF`` assimilator cannot run in NEDAS's
no-mpi4py serial fallback. It does not communicate, though -- PDAF is set up on
``MPI_COMM_SELF`` and every rank analyses the partitions it owns on its own.

PDAF-OMI localizes by horizontal distance alone, so the parts of NEDAS's localization it
has no equivalent for (``vroi``, ``troi`` and ``impact_on_variable``) are refused at
startup rather than silently dropped; use ``ETKF`` for those configurations.

``tests/test_pdaf_letkf.py`` checks PDAF's LETKF analysis against NEDAS's native ETKF on
the same partition, and is skipped unless pyPDAF is installed. The two agree to roundoff
once one difference of convention is accounted for: the localization taper enters PDAF's
analysis linearly (textbook R-localization) and NEDAS's ETKF squared, so the same ``hroi``
gives PDAF a wider effective localization.

PDAF can only be initialized once per process, so the ensemble size, the partitioning and
``assimilator_def.filter_kind`` cannot change within a run; NEDAS raises rather than letting
a second initialization crash the job.

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
