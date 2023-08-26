# NEDAS: The NERSC (New) Ensemble Data Assimilation System

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
or try running the control scripts
`cd scripts; ./run_cycle.sh`


## Documentation

![assimilation work flow](https://github.com/nansencenter/NEDAS/blob/main/tutorials/flowchart.png "Flow chart of one assimilation cycle")
