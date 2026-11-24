#!/usr/bin/env bash
# Build nedas_bgrid_advance: DART's bgrid_solo dry dynamical core (Held-Suarez forcing)
# with a small driver that advances a DART netCDF restart by a given interval (offline io
# mode), and libnedas_bgrid.so, the same model with a C interface for advancing a state in
# memory (online io mode, nedas_bgrid_lib.f90). Both go next to this script.
#
# Uses DART's own source list (as in models/bgrid_solo/work/quickbuild.sh) and mkmf, serial,
# without MPI. Needs gfortran and nf-config. Nothing is written into the DART checkout, which
# needs no site mkmf.template: preprocess, generated sources and all build products stay in ./build.
#
# Usage: ./build_bgrid_solo.sh --dart /path/to/DART [--fc gfortran] [-o OUT]
set -eu

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DART=${DART:-}
fc=gfortran
out="$here/nedas_bgrid_advance"

while [ $# -gt 0 ]; do
  case $1 in
    --dart) DART=$2; shift 2 ;;
    --fc)   fc=$2; shift 2 ;;
    -o)     out=$2; shift 2 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

[ -d "$DART/models/bgrid_solo/fms_src" ] || { echo "ERROR: set --dart to a DART checkout" >&2; exit 1; }
command -v "$fc" >/dev/null || { echo "ERROR: $fc not found" >&2; exit 1; }
command -v nf-config >/dev/null || { echo "ERROR: nf-config not found" >&2; exit 1; }
DART=$(cd "$DART" && pwd)
export DART

build=$here/build
mkdir -p "$build"
cd "$build"
rm -f -- *.o *.mod Makefile.bgrid

nfprefix=$(nf-config --prefix)
ncprefix=$(nc-config --prefix 2>/dev/null || echo "$nfprefix")
cat > mkmf.template.bgrid <<EOF2
FC = $fc
LD = $fc
INCS = -I$nfprefix/include
LIBS = -L$nfprefix/lib -L$ncprefix/lib -lnetcdff -lnetcdf -Wl,-rpath,$nfprefix/lib -Wl,-rpath,$ncprefix/lib
FFLAGS = -O2 -fPIC -ffree-line-length-none \$(INCS)
LDFLAGS = \$(FFLAGS) \$(LIBS)
EOF2

# preprocess: obs types and quantities for the atmosphere, generated files stay in $build
cat > input.nml <<EOF2
&preprocess_nml
   overwrite_output        = .true.
   input_obs_def_mod_file  = '$DART/observations/forward_operators/DEFAULT_obs_def_mod.F90'
   output_obs_def_mod_file = '$build/obs_def_mod.f90'
   input_obs_qty_mod_file  = '$DART/assimilation_code/modules/observations/DEFAULT_obs_kind_mod.F90'
   output_obs_qty_mod_file = '$build/obs_kind_mod.f90'
   obs_type_files          = '$DART/observations/forward_operators/obs_def_reanalysis_bufr_mod.f90'
   quantity_files          = '$DART/assimilation_code/modules/observations/atmosphere_quantities_mod.f90'
   /
&utilities_nml
   /
EOF2

# preprocess itself is built here with our template; DART's buildpreprocess would build it in the
# DART tree with DART's site mkmf.template, which a fresh checkout does not have. With ./preprocess
# present, buildpreprocess below only runs it.
if [ ! -x preprocess ]; then
  mkdir -p pp
  (cd pp && rm -f -- *.o *.mod &&
   "$DART/build_templates/mkmf" -x -t ../mkmf.template.bgrid -m Makefile.pp -p "$build/preprocess" \
       -a "$DART" "$DART/assimilation_code/programs/preprocess/path_names_preprocess" >/dev/null)
fi

extra_list=$DART/models/bgrid_solo/work/extra_source.path_names
set +u
# shellcheck disable=SC1091
source "$DART/build_templates/buildfunctions.sh"
MODEL=bgrid_solo
LOCATION=threed_sphere
EXCLUDE=fms_src
# the model dir holds stand-alone programs next to model_mod.f90; keep them out of the build
model_programs=()
model_serial_programs=(column_rand id_set_def_stdin ps_id_stdin ps_rand_local)
buildpreprocess
arguments nompi
findsrc
set -u

swap() { dartsrc=${dartsrc//$1/$2}; }
swap "$DART/observations/forward_operators/obs_def_mod.f90" "$build/obs_def_mod.f90"
swap "$DART/assimilation_code/modules/observations/obs_kind_mod.f90" "$build/obs_kind_mod.f90"

# shellcheck disable=SC2086
"$DART/build_templates/mkmf" -c "$version_def" -x -a "$DART" \
    -t mkmf.template.bgrid -m Makefile.bgrid -p nedas_bgrid_advance \
    $dartsrc "$extra_list" "$here/nedas_bgrid_advance.f90" "$here/nedas_bgrid_lib.f90"
make -f Makefile.bgrid

[ "$build/nedas_bgrid_advance" -ef "$out" ] || cp "$build/nedas_bgrid_advance" "$out"
echo "built $out"

# the shared library: the same objects, without the program
lib=$(dirname "$out")/libnedas_bgrid.so
objs=$(ls -- *.o | grep -v '^nedas_bgrid_advance\.o$')
# shellcheck disable=SC2086
"$fc" -shared -o "$lib" $objs -L"$nfprefix/lib" -L"$ncprefix/lib" -lnetcdff -lnetcdf \
    -Wl,-rpath,"$nfprefix/lib" -Wl,-rpath,"$ncprefix/lib"
echo "built $lib"
