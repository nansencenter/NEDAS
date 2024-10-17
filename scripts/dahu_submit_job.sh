#!/bin/bash

#OAR -n nedas
#OAR -l /nodes=1/core=4,walltime=00:30:00
#OAR --stdout log/nedas.%jobid%.stdout
#OAR --stderr log/nedas.%jobid%.stdout
#OAR --project pr-sasip
#OAR -t devel

# the stdout and stderr is put into log by default,
# you must create the log directory before running this script

# specify the working directory
WD=/bettik/${USER}

# specify the number of ensemble members
nens=4
# specify the number of processors
nproc=4
# specify the model configuration filename under $WD/NEDAS/samples
model=nextsim.dg

source $HOME/.bashrc
source /applis/environments/conda.sh
#source /bettik/aydogdu-ext/pkgs/nedas-venv/nds/bin/activate
conda activate /bettik/aydogdu-ext/.conda/envs/nedas
export PYTHONPATH=$PYTHONPATH:$WD/nedas-ali

# run nedas experiment
python $WD/nedas-ali/scripts/run_exp.py \
    --config_file=$WD/nedas-ali/config/samples/$model.yml \
    --nens $nens --nproc $nproc \
    --work_dir $WD/DATA/$model # /$(printf "%02d" $nens) #--nproc_mem $SLURM_NTASKS
