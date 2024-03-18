#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_cycle
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --qos=devel
#SBATCH --partition=normal
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/cycle.%j

source ~/.bashrc
##other initial environment src code

##load configuration files, functions, parameters
export config_file=$HOME/code/NEDAS/config/qg_testcase
set -a; source $config_file; set +a

cd $script_dir
source util.sh

trap cleanup SIGINT SIGTERM

if [[ ! -d $work_dir ]]; then mkdir -p $work_dir; fi
cd $work_dir

##list of model components
model_list=`echo "$state_def" |awk '{print $2}' |uniq |sed 's/\./\//g'`

##obs window for all included obs
export obs_window_min=`echo "$obs_def" |awk '{print $4}' |sort -n |head -n1`
export obs_window_max=`echo "$obs_def" |awk '{print $5}' |sort -n |tail -n1`

export obs_time_step_min=`echo $obs_time_steps |awk '{print $1}'`
export obs_time_step_max=`echo $obs_time_steps |awk '{print $NF}'`

#start cycling
date
export time=$time_start
export prev_time=$time
export next_time=$time

while [[ $next_time -le $time_assim_end ]]; do
    export next_time=`advance_time $time $cycle_period`

    echo "--------------------------------------------------"
    echo "current cycle: $time => $next_time"

    if $run_assim && [ $next_time -ge $time_assim_start ] && [ $next_time -le $time_assim_end ]; then
        ##need to run the model until end of obs window
        ##to provide obs priors for assimilation
        export forecast_period=$((cycle_period+$obs_time_step_max+$obs_window_max))
    else
        export forecast_period=$cycle_period
    fi

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
    if $run_assim && [ $time -ge $time_assim_start ] && [ $time -le $time_assim_end ]; then
        $script_dir/module_assim.sh &
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
