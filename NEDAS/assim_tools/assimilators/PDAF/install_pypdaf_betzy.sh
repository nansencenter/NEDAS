#!/usr/bin/env bash
# Build pyPDAF into the active NEDAS conda environment on betzy.
#
# Two betzy-specific details (see install_pypdaf.md):
#  * impi/2021.13, which nedas.src loads, ships gfortran mpi.mod files only up to gfortran
#    11.1 and gfortran 13/14 cannot read those, so the build swaps to impi/2021.15 (up to
#    14.2). Intel MPI holds its ABI across 2021.x, so this stays compatible with the mpi4py
#    in the environment, which is built against 2021.13.
#  * OpenBLAS comes from a module, so it needs an rpath or importing pyPDAF fails once the
#    module is gone.
#
# Usage:  ./install_pypdaf_betzy.sh [/path/to/pyPDAF checkout]
# The checkout must have its PDAF submodule pulled (git submodule update --init --recursive).
set -eu

src=${1:-/cluster/projects/nn2993k/yingyue/code/pyPDAF}
[ -f "$src/meson.build" ] || { echo "ERROR: $src is not a pyPDAF checkout" >&2; exit 1; }
[ -d "$src/PDAF/src" ] || { echo "ERROR: PDAF submodule not pulled in $src -- run:
  (cd $src && git submodule update --init --recursive)" >&2; exit 1; }

source ~/nedas.src
module swap impi impi/2021.15.0-intel-compilers-2025.1.1
module load OpenBLAS/0.3.27-GCC-13.3.0

export FC=mpif90 CC=mpicc      # Intel MPI wrappers around gfortran/gcc
export LDFLAGS="-Wl,-rpath,$EBROOTOPENBLAS/lib ${LDFLAGS:-}"

echo "building pyPDAF from $src (gfortran $(gfortran -dumpversion), $I_MPI_ROOT)"
cd "$src"
pip install . --force-reinstall --no-deps --no-build-isolation \
    -Csetup-args="-Dincdirs=$EBROOTOPENBLAS/include" \
    -Csetup-args="-Dlibdirs=$EBROOTOPENBLAS/lib" \
    -Csetup-args="-Dblas_lib=openblas"

python -c "import pyPDAF; print('pyPDAF ok:', pyPDAF.__file__)"
