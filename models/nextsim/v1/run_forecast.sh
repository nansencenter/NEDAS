#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=nextsim_fcst
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --qos=devel
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/nextsim_fcst.%j

###run deterministic forecast
casename=$1

source ~/.bashrc
##other initial environment src code

##load configuration files, functions, parameters
export config_file=$HOME/code/NEDAS/config/nextsim_testcase
set -a; source $config_file; set +a

cd $script_dir
source util.sh

trap cleanup SIGINT SIGTERM

if [[ ! -d $work_dir ]]; then mkdir -p $work_dir; fi
cd $work_dir

#start cycling
date
export time=$time_start
export prev_time=$time
export next_time=$time

while [[ $next_time -le $time_assim_end ]]; do
    export next_time=`advance_time $time $cycle_period`

    echo "--------------------------------------------------"
    echo "current time step: $time => $next_time"

    export forecast_period=$cycle_period

    ##make the necessary directories
    mkdir -p cycle/$time

    ##clear previous error tags
    for module in `ls cycle/$time/`; do
        stat=cycle/$time/$module/stat
        touch $stat
        if [[ `cat $stat` != "complete" ]]; then
            echo waiting > $stat
        fi
    done

    ###prepare icbc
    $script_dir/../models/nextsim/v1/module_icbc.sh &

    ###model forecast step
    $script_dir/../models/nextsim/v1/module_forecast.sh &
    wait

    ##check errors
    for module in `ls -t cycle/$time/`; do
        stat=cycle/$time/$module/stat
        if [[ `cat $stat` == "error" ]]; then
        echo cycling stopped due to failed module: $module
        exit 1
        fi
    done

    ##advance to next cycle
    export prev_time=$time
    export time=$next_time

done

echo cycling complete

if [[ ! -z $casename ]]; then
    echo moving the output to $workdir/$casename
    mv cycle $casename
fi

