#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=perturb2
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --qos=devel
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/%j

source ~/.bashrc
##other initial environment src code

#load configuration files, functions, parameters
export SCRIPT_DIR=$HOME/code/NEDAS
export CONFIG_FILE=$SCRIPT_DIR/config/test_cases/control
. $CONFIG_FILE
. util.sh

cd $SCRIPT_DIR
. $HOME/python.src; . $HOME/yp/bin/activate
set -a; . $CONFIG_FILE; set +a

total_period=`diff_time $DATE_START $DATE_END`
n_cycle=$((total_period/$CYCLE_PERIOD))

ncnt=0
ntot=$SLURM_NTASKS

cd $SCRIPT_DIR/perturbation
for m in `seq 1 $NUM_ENS`; do
    $SCRIPT_DIR/job_submit.sh 1 1 $ncnt python assemble_pert.py $m $n_cycle &
    ncnt=$((ncnt+1))
    if [[ $ncnt == $ntot ]]; then
        ncnt=0
        wait
    fi
done
wait

cd $SCRIPT_DIR/icbc
for m in `seq 1 $NUM_ENS`; do
    $SCRIPT_DIR/job_submit.sh 1 1 $ncnt python add_pert.py $m &
    ncnt=$((ncnt+1))
    if [[ $ncnt == $ntot ]]; then
        ncnt=0
        wait
    fi
done
wait
