#!/usr/bin/env bash
# Build libdartkernels.so -- DART's serial-filter kernels, C-callable from NEDAS.
#
# Reuses DART's own build machinery (buildfunctions.sh/mkmf) so the source list,
# preprocess step and compiler flags stay whatever upstream says they are; we only
# add -fPIC/-shared on top and compile one extra file (dart_kernels.f90).
#
# Requires: a DART checkout configured the usual way (build_templates/mkmf.template
# in place), the NETCDF env var, and a work directory whose quickbuild.sh names the
# model/location to build against -- any model will do, the kernels themselves are
# model-independent, but assim_tools_mod's dependency chain needs *some* model_mod
# and a preprocess-generated obs_kind_mod.
#
# Usage:
#   export NETCDF=/path/to/netcdf
#   ./build_dart_kernels.sh --dart /path/to/DART [--work-dir DIR] [-o OUT.so]
set -eu

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DART=${DART:-}
work=""
out="$here/libdartkernels.so"

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case $1 in
    --dart)      DART=$2; shift 2 ;;
    --work-dir)  work=$2; shift 2 ;;
    -o)          out=$2;  shift 2 ;;
    -h|--help)   usage 0 ;;
    *) echo "unknown argument: $1" >&2; usage 1 ;;
  esac
done

[ -n "$DART" ] || { echo "ERROR: set \$DART or pass --dart /path/to/DART" >&2; exit 1; }
[ -d "$DART/build_templates" ] || { echo "ERROR: $DART does not look like a DART checkout" >&2; exit 1; }
[ -n "${NETCDF:-}" ] || { echo "ERROR: set \$NETCDF to your netcdf-fortran prefix" >&2; exit 1; }

# DART's own build steps (preprocess included) call mkmf with no -t, so this file has
# to exist -- it is part of setting up a DART checkout, not something we can guess.
template=$DART/build_templates/mkmf.template
[ -f "$template" ] || {
  echo "ERROR: $template not found. Copy the one for your compiler, e.g." >&2
  echo "  cp $DART/build_templates/mkmf.template.gfortran $template" >&2
  exit 1; }

work=${work:-$DART/models/lorenz_96/work}
[ -f "$work/quickbuild.sh" ] || { echo "ERROR: no quickbuild.sh in $work" >&2; exit 1; }

# --- expose the two kernels we call. They are private to assim_tools_mod upstream;
# this adds one `public ::` line and nothing else. Idempotent; `git checkout` reverts it.
assim_tools=$DART/assimilation_code/modules/assimilation/assim_tools_mod.f90
if ! grep -q 'added by NEDAS' "$assim_tools"; then
  echo "patching $assim_tools to expose the obs_increment kernels"
  sed -i '/^public :: filter_assim/i public :: obs_increment_eakf, obs_increment_enkf, obs_increment_kernel, &\n          obs_increment_particle, obs_increment_rank_histogram, obs_increment_gamma, &\n          obs_increment_bounded_norm_rhf, get_truncated_normal_like, update_from_obs_inc, &\n          inc_ran_seq, first_inc_ran_call, assim_tools_init  ! added by NEDAS build_dart_kernels.sh' "$assim_tools"
fi

# --- source list, preprocess and version string all come from DART itself
export DART
MODEL=$(sed -n 's/^MODEL=//p'       "$work/quickbuild.sh" | head -1)
LOCATION=$(sed -n 's/^LOCATION=//p' "$work/quickbuild.sh" | head -1)
[ -n "$MODEL" ] && [ -n "$LOCATION" ] || { echo "ERROR: cannot read MODEL/LOCATION from $work/quickbuild.sh" >&2; exit 1; }
echo "building against model=$MODEL location=$LOCATION (serial, null_mpi)"

# DART's build functions are written for `set -e` alone and read unset positionals,
# so drop -u while we are inside them
set +u
# shellcheck disable=SC1091
source "$DART/build_templates/buildfunctions.sh"

cd "$work"
buildpreprocess          # generates obs_kind_mod.f90 / obs_def_mod.f90
arguments nompi          # null_mpi: NEDAS calls these kernels per-rank from python
findsrc                  # sets $dartsrc
set -u

# -fPIC/-shared on top of whichever template this DART is configured with
cat > mkmf.template.dartkernels <<EOF
include $template
FFLAGS += -fPIC
SHR = -shared
EOF

# shellcheck disable=SC2086
"$DART/build_templates/mkmf" -c "$version_def" -x -a "$DART" \
    -t mkmf.template.dartkernels \
    -m Makefile.dartkernels \
    -p libdartkernels.so \
    $dartsrc \
    "$here/dart_kernels.f90"

mkdir -p "$(dirname "$out")"
[ "$work/libdartkernels.so" -ef "$out" ] || cp "$work/libdartkernels.so" "$out"
echo "built $out"
