#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=forecast
#SBATCH --time=0-04:00:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=128
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/%j

source ~/.bashrc
##other initial environment src code

#load configuration files, functions, parameters
export SCRIPT_DIR=$HOME/code/NEDAS
export CONFIG_FILE=$SCRIPT_DIR/config/test_cases/wind_era5
. $CONFIG_FILE
. util.sh

casename=$EXP_NAME

source $SCRIPT_DIR/env/betzy/nextsim.src

export DATE=$DATE_START

ntot=$SLURM_NNODES
ncnt=0
for m in `seq 1 $NUM_ENS`; do
    mem=`padzero $m 3`
    rundir=/cluster/work/users/yingyue/nextsim_ens_runs/$casename/$mem
    mkdir -p $rundir
    cd $rundir

    #link data files
    rm -rf data
    mkdir data
    cd data
    ln -fs /cluster/projects/nn2993k/sim/data/BATHYMETRY/* .
    ln -fs /cluster/work/users/yingyue/data/TOPAZ4/TP4DAILY_* .

    if [[ $ATMOS_TYPE == "era5" ]]; then
        ln -fs /cluster/work/users/yingyue/data/ERA5/$mem ERA5
    fi
    if [[ $ATMOS_TYPE == "generic_ps" ]]; then
        ln -fs /cluster/work/users/yingyue/data/GENERIC_PS_ATM/$casename/$mem GENERIC_PS_ATM
    fi
    cd ..

    export NEXTSIM_DATA_DIR=`pwd`/data
    $SCRIPT_DIR/forecast_models/nextsim/namelist.sh > nextsim.cfg
    mkdir -p restart
    cd restart
    ln -fs $SCRATCH/nextsim_ens_runs/init_run/restart/{field,mesh}_${DATE:0:8}T${DATE:8:4}00Z.{bin,dat} .
    cd ..
    $SCRIPT_DIR/job_submit.sh 1 128 $ncnt $CODE_DIR/nextsim/model/bin/nextsim.exec --config-files=nextsim.cfg >& out &
    ncnt=$((ncnt+1))
    if [[ $ncnt == $ntot ]]; then
        ncnt=0
        wait
    fi
done
wait
