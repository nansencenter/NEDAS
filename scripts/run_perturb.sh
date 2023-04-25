#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=perturb
#SBATCH --time=0-04:00:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/%j

source ~/.bashrc
##other initial environment src code

#load configuration files, functions, parameters
export SCRIPT_DIR=$HOME/code/NEDAS
export CONFIG_FILE=$SCRIPT_DIR/config/test_cases/control
. $CONFIG_FILE
. util.sh

rundir=$WORK_DIR/run/perturbation
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

#Check dependency

echo running > stat

echo running perturbation...

n_batch=$((NUM_ENS/$PERTURB_NUM_ENS))
total_period=`diff_time $DATE_START $DATE_END`
n_cycle=$((total_period/$CYCLE_PERIOD))

for t in `seq 57 $n_cycle`; do
    ncnt=0
    ntot=40 #$SLURM_NTASKS
    for m in `seq 1 $n_batch`; do
        mkdir -p mem`padzero $m 3`
        cd mem`padzero $m 3`
        for s in `seq 1 $PERTURB_NUM_SCALE`; do
            mkdir -p scale$s
            cd scale$s
            . $SCRIPT_DIR/env/$HOSTTYPE/perturbation.src
            ln -fs $PERTURB_PARAM_DIR/param.sh .
            echo $t $m $s
            $SCRIPT_DIR/perturbation/namelist.sh $s $t > perturbation.nml
            $SCRIPT_DIR/job_submit.sh 1 1 $ncnt $SCRIPT_DIR/perturbation/src/perturbation.exe >& perturbation.log &
            cd ..
            ncnt=$((ncnt+1))
            if [[ $ncnt == $ntot ]]; then
                ncnt=0
                wait
            fi
        done
        cd ..
    done
    wait
done

echo complete > stat
