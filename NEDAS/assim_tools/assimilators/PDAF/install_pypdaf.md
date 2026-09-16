# Installing pyPDAF (for the PDAF assimilator)

pyPDAF has no PyPI or conda-forge package (checked 2026-09-16: `pip install pyPDAF` and
`conda search -c conda-forge pypdaf` both find nothing), so it is built from source against
a PDAF release. The build is meson-python + Cython over PDAF's Fortran library, so it needs
a Fortran compiler, MPI, and BLAS/LAPACK -- the same stack PDAF itself needs.

    git clone --recursive https://github.com/yumengch/pyPDAF.git
    cd pyPDAF
    # edit the compiler/BLAS settings in meson.options for your machine
    pip install .

On betzy, build it inside the NEDAS conda environment (`source ~/nedas.src`) with the
compiler and MKL modules that environment expects loaded, so pyPDAF links against the same
MPI as mpi4py -- a mismatch there shows up as a hang or a crash at the first PDAF call, not
as an import error.

Check the install with

    python -c "import pyPDAF; print(pyPDAF.__file__)"

The assimilator imports pyPDAF lazily, so NEDAS runs fine without it as long as
`assimilator_def.type` is not `PDAF`.
