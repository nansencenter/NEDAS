#!/bin/bash

#OAR -n nedas
#OAR -l /nodes=1/core=8,walltime=00:30:00
#OAR --stdout log/nedas.%jobid%.stdout
#OAR --stderr log/nedas.%jobid%.stdout
#OAR --project pr-sasip
#OAR -t devel

DFLT_USER=yumengch-ext
WD=/bettik/${USER}
CD=${WD}/NEDAS

nens=2
nproc=8
model=nextsim.dg

source $HOME/.bashrc
source /applis/environments/conda.sh
conda activate nedas
export PYTHONPATH=$PYTHONPATH:$WD/NEDAS

sed "s;${DFLT_USER};${USER};g" $CD/config/samples/$model.yml > $CD/config/samples/${model}_${USER}.yml

python $CD/scripts/run_expt.py \
    --config_file=$CD/config/samples/${model}_${USER}.yml \
    --nens $nens \
    --nproc $nproc \
    --work_dir $WD/DATA/$model/ndg_$(printf "%02d" $nens)
