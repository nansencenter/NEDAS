# NEDAS: The NERSC (New) Ensemble Data Assimilation System

initiated by Yue Ying 2022

with contribution from: Anton Korosov, Timothy Williams (pynextsim libraries), NERSC-HYCOM-CICE group (pythonlib for abfile, confmap, etc.), Jiping Xie (enkf-topaz), Tsuyoshi Wakamatsu (BIORAN), Francois Counillon, Yiguo Wang, Tarkeshwar Singh (EnKF in NorCPM)

A unitified ensemble data assimilation scripts/code for NERSC models: HYCOM (TOPAZ), neXtSIM, and (potentially) WRF, ECOSMO, and NorESM. Features: multiscale approach using a common analysis grid with coarse-graining in 4D (x,y,z,t), more flexible localization, inflation for each scale component; a new alignment technique to reduce position/timing mismatch between model forecast and observed features; Overall modular design using Python with MPI parallelization, aiming at scalable algorithms for large dimensional problems.

## Quick Start

Clone the repository at your `$CODE_DIR`:
`git clone git@github.com:nansencenter/NEDAS.git`

Set the python path environment variable, in your .bashrc or other system config files:
`export PYTHONPATH=$PYTHONPATH:$CODE_DIR/NEDAS`

For Linux/MacOS systems:
Edit the configuration files in e.g. config/local/defaults, including directories and experiment setups. You can make a new config file each time you design a new experiment. The config file path is `export CONFIG_FILE=$CODE_DIR/NEDAS/config/local/defaults`.

Go to the `tutorials/` to play with the modules provided by NEDAS.
Before starting a jupyter-notebook or running the python code, source the config file so that the python subprocess gets the setting through system environment variables:
`cd $CODE/NEDAS`
`set -a; source config/local/defaults; set +a`
then
`cd tutorials; jupyter-notebook`
