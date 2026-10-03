# bgrid_solo: DART's dry dynamical core with Held-Suarez forcing

The GFDL FMS B-grid atmospheric dynamical core, as packaged in
[DART](https://github.com/NCAR/DART) (`models/bgrid_solo`): a global dry atmosphere forced by
Newtonian relaxation to a zonally symmetric temperature and Rayleigh friction near the
surface (Held and Suarez 1994). The default grid is 60 x 30 x 5, with prognostic surface
pressure `ps`, temperature `t` and winds `u`, `v`. It develops baroclinic waves and
midlatitude storm tracks, at a cost of about 0.35 s per model day.

## Build

The model source is DART's, it is not copied here. `build_bgrid_solo.sh` compiles it, with
the small driver `nedas_bgrid_advance.f90`, into the `nedas_bgrid_advance` executable:

    ./build_bgrid_solo.sh --dart /path/to/DART

It needs gfortran (or `--fc` another Fortran compiler) and netCDF-Fortran (`nf-config`), no MPI.
DART's own `integrate_model` cannot be used since it only advances to a target time that its
async machinery passes in a file.

## Interface

* Offline io only: the state is a DART netCDF restart file per member and time,
  `<path>/<yyyymmdd_HHMM>[_memNNN].nc`, in the layout of DART's `perfect_input.nc`, so DART's own
  tools can read them. `run` advances a file by `forecast_period` hours (a multiple of `dt_atmos`).
* All variables are on the temperature grid. The wind is on a staggered grid in the model, so reading
  it averages the four surrounding points, and writing it adds the change of the averaged field to
  the wind points. A field that is left unchanged is written back exactly.
* The model starts from its own cold start (at rest, uniform temperature) plus a random temperature
  perturbation, and needs about 100 days (`spinup_hours`) to reach a statistically steady state. The
  truth run is spun up that way. Ensemble members are either independent spun-up cold starts
  (`init_ens_mode: spinup`, a climatological ensemble) or the truth plus a temperature perturbation
  (`init_ens_mode: truth`).

See `default.yml` for the options and `examples/bgrid_solo` for a cycled experiment.
