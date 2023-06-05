# NEDAS: The NERSC (New) Ensemble Data Assimilation System

initiated by Yue Ying 2022

A unitified ensemble data assimilation scripts/code for NERSC models: HYCOM, neXtSIM, WRF, ECOSMO, NorCPM (potentially). Features: common analysis grid; multiscale (localization, inflation, alignment) DA algorithms; modularized design.

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
