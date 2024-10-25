#!/bin/bash

#OAR -n nextsimdg
#OAR -l /nodes=1/core=32,walltime=00:30:00
#OAR --stdout nextsimdg.%jobid%.stdout
#OAR --stderr nextsimdg.%jobid%.stdout
#OAR --project pr-sasip
#OAR -t devel
#OAR --array N
##
. ${HOME}/.bashrc
. /applis/environments/conda.sh
conda activate /bettik/aydogdu-ext/.conda/envs/ndg

export OMP_NUM_THREADS=8

export LD_LIBRARY_PATH=/lib64:/lib/x86_64-linux-gnu:/usr/lib:/home/aydogdu-ext/.conda/envs/ndg/lib
echo $LD_LIBRARY_PATH

echo "starting the script..."

RUN_SCRIPT_DIR=/bettik/aydogdu-ext/NEDAS/models/nextsim/dg
NDG_BLD_DIR=/bettik/aydogdu-ext/nextsimdg/build
NDS_NDG_DIR=/bettik/aydogdu-ext/NEDAS/models/nextsim/dg

INPUT_DATA_DIR=/summer/sasip/model-forcings/nextsim-dg
SCRATCH=$(pwd)
# go to nextsim-dg directory
#[ ! -z ${SCRATCH} ] && mkdir -p ${SCRATCH} && cd ${SCRATCH}
cd ${SCRATCH}
# change to ensemble directory 
cd ens_$(printf "%02d" ${OAR_ARRAY_INDEX})

# link to the executable you've just compiled
cp ${NDG_BLD_DIR}/nextsim .
cp ${NDS_NDG_DIR}/default.cfg ndg.cfg

# run the model
./nextsim --config-file ndg.cfg > time.step
