#!/bin/bash

#OAR -n nedas
#OAR -l /nodes=1/core=32,walltime=00:30:00
#OAR --stdout log/nedas.%jobid%.stdout
#OAR --stderr log/nedas.%jobid%.stdout
#OAR --project pr-sasip
#OAR -t devel

WD=/bettik/${USER}
CD=${WD}/NEDAS

nens=2
nproc=8
model=nextsim.dg

source $HOME/.bashrc
source /applis/environments/conda.sh
#source /bettik/aydogdu-ext/pkgs/nedas-venv/nds/bin/activate
conda activate nedas
export PYTHONPATH=$PYTHONPATH:$WD/NEDAS

python $CD/scripts/run_exp.py --config_file=$CD/config/samples/$model.yml --nens $nens --nproc $nproc --work_dir $WD/DATA/$model/ndg_$(printf "%02d" $nens)
