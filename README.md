# NEDAS: The Next-generation Ensemble Data Assimilation System

A unitified ensemble data assimilation scripts/code for NERSC models: HYCOM (TOPAZ), neXtSIM, and (potentially) WRF, ECOSMO, and NorESM. Features: multiscale approach using a common analysis grid with coarse-graining in 4D (x,y,z,t), more flexible localization, inflation for each scale component; a new alignment technique to reduce position/timing mismatch between model forecast and observed features; Overall modular design using Python with MPI parallelization, aiming at scalable algorithms for large dimensional problems.

initiated by Yue Ying 2022

with contribution from: Anton Korosov, Timothy Williams (pynextsim libraries), NERSC-HYCOM-CICE group (pythonlib for abfile, confmap, etc.), Jiping Xie (enkf-topaz), Tsuyoshi Wakamatsu (BIORAN), Francois Counillon, Yiguo Wang, Tarkeshwar Singh (EnKF in NorCPM)

## Quick Start

Clone the repository at your `$CODE`:
`git clone git@github.com:nansencenter/NEDAS.git`

Set the python path environment variable, in your .bashrc or other system config files:
`export PYTHONPATH=$PYTHONPATH:$CODE/NEDAS`

Go to `config/defaults` for a list of environment variables, change accordingly. Save your own configuration in a separate file (one for each experiment) and `CONFIG_FILE` points to its location.

Go to the `tutorials/` to play with the modules provided by NEDAS.
Before starting a jupyter-notebook or running `scripts/*.py`, source the config file so that the python subprocess gets the setting through system environment variables (in bash `set -a` make the env available for subprocesses):
`cd $CODE/NEDAS`
`set -a; source $CONFIG_FILE`
then
`cd tutorials; jupyter-notebook`
or try running the control scripts using one of the test configs in tutorials
`cd scripts; ./run_cycle.sh`


## Documentation

### Code structure:

**scripts** contains bash runtime control scripts `run_cycle.sh`, `run_forecast.sh` for a typical cycling DA experiment; misc. initialisation scripts `prepare_*.sh`; and misc. postprocessing and diagnosis scripts `diag_*.sh`.

**assim\_tools** contains the top-level routines `process_state`, `process_obs`, `local_analysis` and `update_restart` for one assimilation cycle, along with some utility functions `field_info`, `obs_info`, `read_*`, `write_*`, etc.

**grid** contains a `Grid` class to handle the conversion between 2D fields.

**models** contains modules for each forecast model, each with a set of standard routines to process model restart files into state variables: `read_var`, `read_grid`, `z_coords`, etc.

**dataset** contains modules for each dataset file, either providing observations or boundary conditions for model runs. For obs provider modules, the `read_obs` routine converts the raw data to the observation sequence.

**perturb** contains routines for generating initial and boundary perturbations for ensemble model runs.

**diag** contains routines for computing misc. diagnostics for model forecast verification and filter performance evaluation.

**config** contains shell environment src files for different HPCs, and configuration files for setting up a particular experiment (`defaults` provides a sample).

**tutorials** contains a set of jupyter notebooks to demonstrate the functionalities provided by the system.

### Flow chart of one assimilation cycle:
![](https://github.com/nansencenter/NEDAS/blob/main/tutorials/imgs/flowchart.png "Flow chart of one assimilation cycle")


