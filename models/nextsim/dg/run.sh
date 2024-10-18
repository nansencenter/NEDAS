#!/bin/bash

##OAR -n nextsimdg
##OAR -l /nodes=1/core=32,walltime=00:30:00
##OAR --stdout nextsimdg.%jobid%.stdout
##OAR --stderr nextsimdg.%jobid%.stdout
##OAR --project pr-sasip
##OAR -t devel
##
. ${HOME}/.bashrc
. /applis/environments/conda.sh
conda activate ndg

export OMP_NUM_THREADS=8

export LD_LIBRARY_PATH=/lib64:/lib/x86_64-linux-gnu:/usr/lib:/home/aydogdu-ext/.conda/envs/ndg/lib
echo $LD_LIBRARY_PATH

echo "starting the script..."

RUN_SCRIPT_DIR=/bettik/aydogdu-ext/NEDAS/models/nextsim/dg
NDG_BLD_DIR=/bettik/aydogdu-ext/nextsimdg/build
NDS_NDG_DIR=/bettik/aydogdu-ext/NEDAS/models/nextsim/dg

INPUT_DATA_DIR=/summer/sasip/model-forcings/nextsim-dg
SCRATCH=$1

# go to nextsim-dg directory
#[ ! -z ${SCRATCH} ] && mkdir -p ${SCRATCH} && cd ${SCRATCH}
cd ${SCRATCH}

# link to the executable you've just compiled
ln -sf ${NDG_BLD_DIR}/nextsim .
ln -sf ${NDS_NDG_DIR}/default.cfg ndg.cfg

# link to the initial state and forcing files
ln -sf ${INPUT_DATA_DIR}/init_25km_NH.nc .
ln -sf ${INPUT_DATA_DIR}/25km_NH.ERA5_2010-01-01_2011-01-01.nc .
ln -sf ${INPUT_DATA_DIR}/25km_NH.TOPAZ4_2010-01-01_2011-01-01.nc .

# run the model
./nextsim --config-file ndg.cfg > time.step
