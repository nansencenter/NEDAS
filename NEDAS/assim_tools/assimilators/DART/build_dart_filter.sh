#!/usr/bin/env bash
# Build libdartfilter.so: DART's filter_main with its file I/O redirected to NEDAS memory.
#
# Uses DART's own source list and preprocess; swaps in a patched filter_mod.f90
# (patch_filter_mod.py), the NEDAS model_mod and NEDAS obs types/quantities.
# The DART checkout is not modified.
#
# Needs an MPI Fortran compiler matching the MPI that NEDAS's mpi4py uses, and nf-config.
# Betzy: module load netCDF-Fortran/4.6.1-iimpi-2024a  (impi 2021.13, as in nedas.src)
#
# Usage: ./build_dart_filter.sh --dart /path/to/DART [--fc mpiifx] [-o OUT.so]
set -eu

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DART=${DART:-}
mpifc=mpiifx
out="$here/libdartfilter.so"

while [ $# -gt 0 ]; do
  case $1 in
    --dart) DART=$2; shift 2 ;;
    --fc)   mpifc=$2; shift 2 ;;
    -o)     out=$2; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -d "$DART/build_templates" ] || { echo "ERROR: set --dart to a DART checkout" >&2; exit 1; }
command -v "$mpifc" >/dev/null || { echo "ERROR: $mpifc not found" >&2; exit 1; }
command -v nf-config >/dev/null || { echo "ERROR: nf-config not found" >&2; exit 1; }

build=$here/build
mkdir -p "$build"
cd "$build"
rm -f -- *.o *.mod Makefile.dartfilter

# preprocess: NEDAS obs types and quantities, generated files stay in $build
cat > input.nml <<EOF
&preprocess_nml
   overwrite_output        = .true.
   input_obs_def_mod_file  = '$DART/observations/forward_operators/DEFAULT_obs_def_mod.F90'
   output_obs_def_mod_file = '$build/obs_def_mod.f90'
   input_obs_qty_mod_file  = '$DART/assimilation_code/modules/observations/DEFAULT_obs_kind_mod.F90'
   output_obs_qty_mod_file = '$build/obs_kind_mod.f90'
   obs_type_files          = '$here/obs_def_nedas_mod.f90'
   quantity_files          = '$DART/assimilation_code/modules/observations/default_quantities_mod.f90',
                             '$here/nedas_quantities_mod.f90'
   /
&utilities_nml
   /
EOF

python3 "$here/patch_filter_mod.py" \
    "$DART/assimilation_code/modules/assimilation/filter_mod.f90" "$build/filter_mod.f90"
python3 "$here/patch_filter_mod.py" \
    "$DART/assimilation_code/modules/assimilation/algorithm_info_mod.f90" "$build/algorithm_info_mod.f90"

nfprefix=$(nf-config --prefix)
ncprefix=$(nc-config --prefix 2>/dev/null || echo "$nfprefix")
# RPATH (not RUNPATH) so these win over LD_LIBRARY_PATH at load time
rpath="-Wl,--disable-new-dtags -Wl,-rpath,$nfprefix/lib -Wl,-rpath,$ncprefix/lib"
[ -n "${EBROOTHDF5:-}" ] && rpath="$rpath -Wl,-rpath,$EBROOTHDF5/lib"
# link with the wrapper's own command minus its --enable-new-dtags, which would undo the RPATH
mpild=$("$mpifc" -show | sed 's/-Xlinker --enable-new-dtags//g')
cat > mkmf.template.dartfilter <<EOF
MPIFC = $mpifc
MPILD = $mpild
FC = $mpifc
LD = $mpifc
INCS = -I$nfprefix/include
LIBS = -L$nfprefix/lib -L$ncprefix/lib -Wl,--no-as-needed -lnetcdff -lnetcdf $rpath
FFLAGS = -O2 -fPIC \$(INCS)
LDFLAGS = \$(FFLAGS) \$(LIBS)
SHR = -shared
EOF

export DART
MODEL=null_model
LOCATION=threed_cartesian
set +u
# shellcheck disable=SC1091
source "$DART/build_templates/buildfunctions.sh"
buildpreprocess
arguments mpi
findsrc
set -u

swap() { dartsrc=${dartsrc//$1/$2}; }
swap "$DART/observations/forward_operators/obs_def_mod.f90" "$build/obs_def_mod.f90"
swap "$DART/assimilation_code/modules/observations/obs_kind_mod.f90" "$build/obs_kind_mod.f90"
swap "$DART/assimilation_code/modules/assimilation/filter_mod.f90" "$build/filter_mod.f90"
swap "$DART/assimilation_code/modules/assimilation/algorithm_info_mod.f90" "$build/algorithm_info_mod.f90"
swap "$DART/models/null_model/model_mod.f90" "$here/model_mod.f90"

# shellcheck disable=SC2086
"$DART/build_templates/mkmf" -c "$version_def" -x -a "$DART" $m \
    -t mkmf.template.dartfilter -m Makefile.dartfilter -p libdartfilter.so \
    $dartsrc "$here/nedas_hooks_mod.f90" "$here/nedas_filter_api.f90"

[ "$build/libdartfilter.so" -ef "$out" ] || cp "$build/libdartfilter.so" "$out"
echo "built $out"
