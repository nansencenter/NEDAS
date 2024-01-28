#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_cycle
#SBATCH --time=0-00:30:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=125
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/run_cycle.%j

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

##list of model components
model_list=`echo "$state_def" |awk '{print $2}' |uniq |sed 's/\./\//g'`

#start cycling
date
export time=$time_start
export prev_time=$time
export next_time=$time

while [[ $next_time -le $time_assim_end ]]; do
    export next_time=`advance_time $time $cycle_period`

    echo "--------------------------------------------------"
    echo "current cycle: $time => $next_time"

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

    ###prepare icbc and perturb members
    for model in $model_list; do
        $script_dir/../models/$model/module_icbc.sh &
        $script_dir/../models/$model/module_perturb.sh &
    done

    ###data assimilation step
    if [ $time -ge $time_assim_start ] && [ $time -le $time_assim_end ]; then
        if $run_assim; then
            $script_dir/module_assim.sh &
        fi
    fi

    ###model forecast step
    for model in $model_list; do
        $script_dir/../models/$model/module_ens_forecast.sh &
    done
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
