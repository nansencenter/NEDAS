# Changelog

All notable changes to NEDAS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Patch releases (`x.y.Z`) are cut on demand whenever bug fixes accumulate on `develop`,
rather than on a fixed schedule, and contain fixes only — no new features. New models,
DA schemes, or other backward-compatible features land in the next minor release.

## [Unreleased]

### Added
- `ice_conc`, `ice_drift`, and `cs2smos` sea ice datasets: opt-in
  `use_dataset_uncertainty` (per-pixel uncertainty from the source file) and
  `use_adaptive_err` (concentration-/displacement-/thickness-dependent error
  formulas ported from `enkf-topaz`) toggles; both default to `False`, no
  behavior change unless enabled in `dataset_def`
- `InterpolationAssimilator`: Cressman/OI local obs-only analysis (`interp`
  assimilator type)
- `vort3d` obs: core-biased radial sampling option for targeted obs networks
- DIS optical flow: `variational_refine_alpha` now configurable
- `vort3d`: adaptive dt retry on NaN blowup (`dt_reduction_factor`,
  `max_dt_retries`, `min_dt`)

### Fixed
- Pylance/pyright type-checking cleanup across several modules (None-guards,
  numpy-scalar casts, `getattr` over `hasattr`+attribute access); includes an
  `nx`/`ny` swap bug in `vort3d_obs.vortex_position`'s no-vorticity fallback
  that mis-anchored the search window on non-square grids

## [1.3.1] - 2026-08-06

### Fixed
- `BatchAssimilator` localization prefilter: L1 (Manhattan) distance check for the
  `hroi` obs filter now scaled by √2, making the prefilter disk a correct superset
  of the true L2 Gaspari-Cohn support — fixes hard diamond-shaped edges in the
  analysis from obs incorrectly excluded along diagonal directions
- `cs2smos_obs`: netCDF4 fill-value leak — `_FillValue` sentinel was read as a
  literal observation for every no-data pixel (~90% of one test-case grid),
  silently folding the sentinel value into the analysis; now uses `filled(nan)`
  with an explicit finite-validity check before the obs sequence
- `topaz5model`: `hice_impact` (renamed to `seaice_thick_obs_impact`) is now wired
  through `postprocess()` into the external `fixhycom` binary call, which
  previously hardcoded a literal 0 for that argument — the SIT analysis increment
  was computed by the EnKF but never redistributed into the restart's per-category
  ice volume regardless of the configured value. Default remains 0 (no behavior
  change unless explicitly raised)
- `TopazDEnKF`: align with Fortran reference `enkf-topaz` (develop @ 0f4c74b):
  - `rfactor1` parameter added: global obs error inflation factor matching
    Fortran's `RFACTOR1` (applied before local analysis)
  - `kfactor` moved from inside `ensemble_transform_weights()` to a one-time
    adjustment in `local_analysis()` before the field loop, matching Fortran's
    `obs_QC()` flow (applied once, not re-computed per field)
  - `nlobs_max` default changed from 2848 to 0 (no limit), matching Fortran's
    default (`nlobs = 0` → use all obs within localization radius)
  - (Inflation formula verified: Fortran's `infl`-based matrix `IM` produces
    standard multiplicative inflation, *not* relaxation-to-prior — no change
    required on the NEDAS side)
- `Vort3DObs.__init__` `KeyError` when the `vort3d` model is not registered in the
  context (hit by the generic dataset smoke test)
- Sphinx `release`/`version` now derived from `NEDAS.__version__` instead of being
  hardcoded in `docs/conf.py`
- `topaz5model.postprocess`: `fixhycom` iced input now links to the posterior state,
  not the background
- `alignment_updator`: target level `k` is now configurable; added `vector_image`
  option for vector-to-scalar conversion
- Adaptive posterior inflation for multiscale (`once_after_outer_loop` timing):
  configurable `max_coef` cap, and a file-lock re-initialization bug fix
- `qg/python`: bottom/top Ekman drag now uses time-lagged `psi_o` (matching Fortran
  `qg_driver.f90::Get_rhs`) instead of current `psi`; multi-layer spectral initial
  condition now matches the Fortran model's modal-to-layer projection, so results
  reproduce the Fortran version closely
- `ice_drift` obs operator: `iced`/`seaice_velocity` restart lookup now rounds the
  obs valid time to the start of day before the `iced` attempt, matching the
  day-level tolerance `iceh` already had; previously the exact-hour requirement
  for `iced_variables` could never be met (restarts are only ever written at hour
  0), so the lookup silently fell through to the `iceh` fallback, which also fails
  for a cycle prior state before any forecast has run
- `vort3d` obs: fixed cyclic-boundary wrap bias in `vortex_position` centroid
  (tracks no longer pin at the domain wrap point)
- `vort3d` obs: `vortex_position` search window now respects `grid.cyclic_dim`
  per axis (wrap vs. reflect-pad), generalizing the wrap/wall fixes above
- `vort3d` obs: proximity-taper the vorticity centroid toward the anchored
  vortex, so tracks stop jumping to unrelated blobs or pinning at walls
- Seeded obs network/noise RNG by cycle time (and `obs_rec_id`) so synthetic
  obs are reproducible across schemes at a given cycle
- Separated obs error inflation (tempering) from obs generation noise, so
  inflating R for tempering no longer corrupts the generated obs values
- Moved `character_length` into `scale_bandpass`'s own `transform_def` scope;
  iterations without scale decomposition no longer assume it exists
- `vort3d`: extended NaN detection to all prognostic fields (previously only
  `u`/`pstar`) and clip qsat iterates inside the loop, not just after

## [1.3.0] - 2026-07-23

### Added
- `vort3d` model: minimal 3D tropical cyclone model (Zhu, Smith & Ulrich 2001), with
  configurable vertical levels, vortex position/intensity/size, steering flow, f0,
  Betts-Miller closure, and boundary-layer thermodynamic ensemble spread
- `qg/python`: pure-Python QG model backend
- `lorenz96/tracer_advection`: tracer advection extension for the Lorenz-96 model
- `QCEF` assimilator module
- `alignment_updator`: multiscale alignment update scheme
- Simultaneous state-parameter estimation (SSPE)
- Inflation timing option for once-after-outer-loop application
- One-time perturbation support
- Ensemble verification metrics (`diag/metrics/spectral.py`)
- TOPAZ5 archm variables write-back
- DIS/Farneback optical flow: tunable parameters, `local_weight` self-normalizing
  smoothness-weight alternative

### Changed
- **Breaking:** `impact_on_state` renamed to `impact_on_variable` in configs, docs,
  and tests (extended to obs variables); old key silently no-ops rather than erroring
- Per-iteration parameters (`hroi`, `err.std`, updator/assimilator/inflation/transform_def)
  redesigned as explicit `iterN` dicts
- Synthetic obs (network + noise draw) now cached once per analysis cycle instead of
  per outer-loop iteration
- Deferred `xarray`/`scipy`/`matplotlib` imports to avoid slow I/O-bound loading

### Fixed
- `Context.logger()` not restoring `pid_show` around wrapped calls
- `Progress.update()` `ZeroDivisionError` when a rank has zero assigned tasks
- `MultiplicativeInflation.apply_inflation` not restoring `c.pid_show`
- `Vort2DModel`/`Lorenz96Model` memory dict shared across all instances (class-level
  mutable default)
- ETKF `apply_ensemble_transform` renormalizes weight columns instead of aborting
- `obs.py` `state_to_obs` ignoring tag for custom obs_operator
- `AlignmentPreconditioner.warp()` unit mismatch (meters vs. grid-index)
- `obs_prior` output in npy files
- `obs_post` recomputation restricted to batch assimilators, now applies to serial too
- Alignment updator re-warping already-finalized scales instead of freezing them
- Adaptive posterior inflation formula corrected to match Ying (2019) exactly
- DIS/Farneback shared-range uint8 normalization; Horn-Schunck weight now derived
  per-call from alpha_squared
- `filter.prepare_init_ensemble` bug
- Typing errors
- `H(X)` using scale-decomposed field instead of full state in multiscale DA
- `taper_boundary` not available by default
- `Model` missing `params` annotation; `nextsim` `write_param` renamed param->value
- Slurm submitter: `file_pointer` init moved before job-array branch
- Diag plot module crashes and design issues
- `rec.nobs` not synced to all MPI ranks after `prepare_obs` broadcast
- `impact_on_state` cross-variable localization not implemented in EAKF
- Offline scheduler `nworker` calculation on HPC mode (#19)
- Removed unused TensorFlow dependency; PyTorch only for ML (#22)
- Per-variable `dt` time loop in perturb step; MPI-safe file-lock init (#24)
- `@njit` removed from KDE dict/lambda functions; QCEF tests unskipped
- Explicit cycling config to fix `restart_dir` logic for offline DA
- MPI ranks agreeing on `obs_post` availability before allreduce (inflation)
- Lorenz-96 bug fix
- `prepare_init_ensemble`/`ensemble_forecast` now use the full `nproc` budget

### Removed
- Preconditioner (reverted)

## [1.2.0] - 2026-04-22

### Added
- New `core` module consolidating the `Model`, `Dataset`, and `Scheme` base classes
  (previously scattered across submodules), including a dedicated `Context` class
  to hold runtime-living objects (previously done by `Config`)
- `IOBackend` classes implementing both online and offline I/O modes; online mode
  adds a memory save/load mechanism and is supported by the `lorenz96` and `vort2d`
  (native Python) models
- Generalized `Grid` class hierarchy (`Grid1D`, `RegularGrid`, `IrregularGrid`)
- `Progress` class for runtime logging: interactive on/off modes, terminal-size
  detection, Jupyter notebook support
- AMSR2 dataset: SIC retrieval and obs_operator
- CS2SMOS sea-ice thickness dataset
- New synthetic-obs subclass with prescribed `obs_x`/`obs_y`/`obs_z` support, and
  `save_obs` to keep a copy of `obs_seq` in `dataset.memory`
- `filter` and `forecast` (forecast-only) analysis schemes
- TOPAZ5: `write_var` for `iceh` variables, `tcwv`/`tclw` made operational in
  preprocessing, updated namelist and `iage` variable, brightness-temperature (Tb)
  assimilation support, conc reading from either `iced` or `iceh` files
- Automatic package versioning from git tags (setuptools_scm)
- Call-stack management, throttled `Progress.update()`, persistent logger
  parameters, and memory dump-to-file in the runtime logger

### Changed
- `assim_tools`, perturbation, and scheme base classes substantially refactored
- `qg` model renamed to `qg.fortran` to make room for future backends
- Config key `analysis_scheme` renamed to `scheme`; empty `model`/`dataset` config
  entries are now allowed; config objects can be kept read-only
- `model.variables` changed to `dict[str, VarDesc]`
- Type hints made backward-compatible with Python 3.9

### Fixed
- Divide-by-zero in `gaspari_cohn`
- `obs_prior` NaN values not filtered out in the batch assimilator
- Parallel finalization safety issues; nproc/nproc_util ambiguity resolved
- AMSR2 channel-order bug; OSISAF `ice_drift` grid definition and time-window
  mismatch; OSISAF `ice_conc` filename time-string bug
- Random perturbation multiscale bug
- `pandas`/`xarray` made required dependencies (fixed missing-dependency crashes)
- GridType checks; job submitter fixes for SLURM/OAR/gricad, including kill-signal
  handling and clearer error messages
- Large batch of pylint and typing cleanups across the codebase

## [1.1.0] - 2025-06-24

An interim `1.0.1` milestone (workflow rewritten in Python, never separately
tagged) landed between `1.0-beta` and this release — folded in below.

### Added
- `assim_tools` refactored into composable component classes:
  **Assimilators** (ETKF, TopazDEnKF, EAKF), **Updators** (Additive, Alignment),
  and transform functions (null, scale_bandpass)
- New models: `lorenz96`, `nextsim/v1`, `nextsim/dg`, `noresm`, `qg` (+ emulator),
  `topaz/v4`, `topaz/v5`, `vort2d`, `wrf`
- New datasets: `era5`, `argo` (ifremer), `rgps`, `osisaf` (ice_conc, ice_drift),
  `qg`, `lorenz96`, `vort2d`
- Adaptive inflation algorithms
- readthedocs documentation site
- QG-model benchmark example comparing filter/DA algorithm performance
- Published to PyPI

### Changed
- Configuration is now fully YAML-file-based; no more Linux environment variables
- Models are now `Model` classes with user-provided methods (`read_var`, etc.)
  instead of shell-script modules
- Ensemble forecast runs in either batch mode or via a job scheduler
- Workflow control moved to Python (`scripts/run_exp.py` as top-level entry point,
  with `assimilate.py` and `ensemble_forecast.py` as the two main steps)

## [1.0-beta] - 2024-01-17

Initial public beta release.

### Added
- Bash-script-driven workflow control, with model code run via
  `model/<model>/module_forecast.sh`
- Parallel DA step via `scripts/run_assim.py`
- Configuration via environment variables defined in `config/*`
- Demo cases for the `vort2d` and `qg` models; a `qg` benchmark for comparing DA
  algorithm efficiency

[Unreleased]: https://github.com/nansencenter/NEDAS/compare/v1.3.0...HEAD
[1.3.0]: https://github.com/nansencenter/NEDAS/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/nansencenter/NEDAS/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/nansencenter/NEDAS/compare/v1.0-beta...v1.1.0
[1.0-beta]: https://github.com/nansencenter/NEDAS/releases/tag/v1.0-beta
