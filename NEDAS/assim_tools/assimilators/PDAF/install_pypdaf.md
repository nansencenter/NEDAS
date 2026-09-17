# Installing pyPDAF (for the PDAF assimilator)

pyPDAF has no PyPI or conda-forge package (checked 2026-09-16: `pip install pyPDAF` and
`conda search -c conda-forge pypdaf` both find nothing), so it is built from source against a
PDAF release. The build is meson-python + Cython over PDAF's Fortran library, so it needs a
Fortran compiler, MPI and BLAS/LAPACK -- the same stack PDAF itself needs.

    git clone --recursive https://github.com/yumengch/pyPDAF.git   # --recursive: PDAF is a submodule
    cd pyPDAF
    FC=<mpi fortran wrapper> CC=<mpi c wrapper> pip install . --no-build-isolation \
        -Csetup-args="-Dincdirs=<blas include>" \
        -Csetup-args="-Dlibdirs=<blas lib>" \
        -Csetup-args="-Dblas_lib=openblas"

meson does not look for MPI itself on Linux (`dependency('', required: false)`), so MPI comes
entirely from the compiler wrapper named in `FC`/`CC`.

## On betzy

Built into the NEDAS conda env (2026-09-17) with `install_pypdaf_betzy.sh` (next to this file), which is the
recipe above plus two things worth knowing:

* **The MPI module has to be swapped.** `impi/2021.13`, which `nedas.src` loads, ships
  gfortran `mpi.mod` files only up to gfortran 11.1, and gfortran 13/14 cannot read those
  (`Fatal Error: Reading module ... Unexpected EOF`). `impi/2021.15` ships them up to 14.2.
  Intel MPI keeps its ABI across the 2021.x series, so the result still runs against the
  mpi4py in the env, which is built against 2021.13.
* **OpenBLAS needs an rpath.** `-Wl,-rpath,$EBROOTOPENBLAS/lib`, or importing pyPDAF fails
  with `libopenblas.so.0: cannot open shared object file` unless the module is loaded.

To run anything that calls PDAF on a *login* node, `unset I_MPI_PMI_LIBRARY` first: PDAF calls
MPI_Init itself, and Intel MPI then tries SLURM's PMI2 and aborts with `PMI2_Job_GetId
returned 14`. Inside a job step there is nothing to unset.

Check the install with

    python -c "import pyPDAF; print(pyPDAF.__file__)"

The assimilator imports pyPDAF lazily, so NEDAS runs fine without it as long as
`assimilator_def.type` is not `PDAF`.

## MPI is not optional

pyPDAF requires MPI at build time (`dependency('mpi', ..., required: true)` on Windows, an MPI
wrapper compiler elsewhere) and initializes it on import, so unlike the DART backend (built
`nompi`) the PDAF assimilator cannot run in NEDAS's no-mpi4py serial fallback. It does not
*communicate*, though: PDAF is set up on MPI_COMM_SELF and every rank analyses its own
partitions alone.
