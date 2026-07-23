# Changelog

All notable changes to NEDAS are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Patch releases (`x.y.Z`) are cut on demand whenever bug fixes accumulate on `develop`,
rather than on a fixed schedule, and contain fixes only — no new features. New models,
DA schemes, or other backward-compatible features land in the next minor release.

## [Unreleased]

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
  (previously scattered across submodules)
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
- Online/offline I/O backend logic extracted out of the model classes and
  refactored into `io_backend`
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
