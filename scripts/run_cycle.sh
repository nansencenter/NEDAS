#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_cycle
#SBATCH --time=0-00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --qos=devel
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/%j

source ~/.bashrc
##other initial environment src code

#load configuration files, functions, parameters
export script_dir=$HOME/code/NEDAS/scripts
export config_file=$HOME/code/NEDAS/config/defaults
. $config_file

cd $script_dir
. util.sh

#start cycling
date
export time=$time_start
export prev_time=$time
export next_time=$time

while [[ $next_time -le $time_assim_end ]]; do  #CYCLE LOOP
    export next_time=`advance_time $time $cycle_period`

    echo "----------------------------------------------------------------------"
    echo "current cycle: $time => $next_time"
    mkdir -p $WORK_DIR/{forecast,analysis}/$time

    ##clear previous error tags
    for d in `ls analysis/$time/`; do
        if [[ `cat run/$DATE/$d/stat` != "complete" ]]; then
        echo waiting > run/$DATE/$d/stat
        fi
    done

    ###run components---------------------------------------

    ###icbc
    $script_dir/module_icbc.sh &
    $script_dir/module_gen_perturbation.sh &

    ###data assimilation step
    if [ $DATE -ge $DATE_CYCLE_START ] && [ $DATE -le $DATE_CYCLE_END ]; then
        if $RUN_ENKF; then
            $script_dir/module_filter_update.sh &
        fi
    fi
    wait

    ###model forecast step
    $script_dir/module_forecast.sh &
    wait


    ###check errors
    for d in `ls -t run/$time/`; do
        if [[ `cat run/$time/$d/stat` == "error" ]]; then
        echo cycling stopped due to failed component: $d
        exit 1
        fi
    done

    ###advance to next cycle
    export prev_time=$time
    export time=$next_time
done
echo cycling complete
