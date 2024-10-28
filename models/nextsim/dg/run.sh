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
conda activate nextsimdg

export OMP_NUM_THREADS=8

echo "starting the script..."

NDG_BLD_DIR=/bettik/yumengch-ext/nextsimdg/build
NDS_NDG_DIR=/bettik/yumengch-ext/NEDAS/models/nextsim/dg

INPUT_DATA_DIR=/summer/sasip/model-forcings/nextsim-dg
SCRATCH=$(pwd)
# go to nextsim-dg directory
#[ ! -z ${SCRATCH} ] && mkdir -p ${SCRATCH} && cd ${SCRATCH}
cd ${SCRATCH}
echo ${SCRATCH}
cd ens_$(printf "%02d" ${OAR_ARRAY_INDEX})

echo $(pwd)
# link to the executable you've just compiled
cp ${NDG_BLD_DIR}/nextsim .
cp ../default.cfg ndg.cfg

# run the model
./nextsim --config-file ndg.cfg > time.step


