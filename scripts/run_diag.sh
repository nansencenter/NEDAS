#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=diag
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=preproc
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

cd $SCRIPT_DIR/diag
for m in `seq 1 $NUM_ENS`; do
    for v in 0 1; do
        srun -n 1 -N 1 --exact python process_forcing.py $SCRATCH/nextsim_ens_runs/wind10m_err1.0/working/`padzero $m 3` $v &
        ncnt=$((ncnt+1))
        if [[ $ncnt == $ntot ]]; then
            ncnt=0
            wait
        fi
    done
done

