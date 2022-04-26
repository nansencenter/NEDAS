#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=NEDAS_run_cycle
#SBATCH --time=0-00:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --qos=devel

source ~/.bashrc
##other initial environment src code

#load configuration files, functions, parameters
export SCRIPT_DIR=$HOME/code/NEDAS
cd $SCRIPT_DIR
export CONFIG_FILE=$SCRIPT_DIR/config/defaults
. $CONFIG_FILE
. util.sh

#start cycling
date
export DATE=$DATE_START
export PREVDATE=$DATE
export NEXTDATE=$DATE

while [[ $NEXTDATE -le $DATE_CYCLE_END ]]; do  #CYCLE LOOP
    export NEXTDATE=`advance_time $DATE $CYCLE_PERIOD`

    echo "----------------------------------------------------------------------"
    echo "CURRENT CYCLE: $DATE => $NEXTDATE"
    mkdir -p $WORK_DIR/{run,ens,diag,obs}/$DATE

    #CLEAR ERROR TAGS
    for d in `ls run/$DATE/`; do
        if [[ `cat run/$DATE/$d/stat` != "complete" ]]; then
        echo waiting > run/$DATE/$d/stat
        fi
    done

    #RUN COMPONENTS---------------------------------------

    # ICBC
    $SCRIPT_DIR/module_icbc.sh &
    $SCRIPT_DIR/module_gen_perturbation.sh &

    # Data assimilation step
    if [ $DATE -ge $DATE_CYCLE_START ] && [ $DATE -le $DATE_CYCLE_END ]; then
        if $RUN_ENKF; then
            $SCRIPT_DIR/module_filter_update.sh &
        fi
    fi
    wait

    # Forecast step
    $SCRIPT_DIR/module_forecast.sh &
    wait


    #CHECK ERRORS
    for d in `ls -t run/$DATE/`; do
        if [[ `cat run/$DATE/$d/stat` == "error" ]]; then
        echo CYCLING STOP DUE TO FAILED COMPONENT: $d
        exit 1
        fi
    done

    #ADVANCE TO NEXT CYCLE
    export PREVDATE=$DATE
    export DATE=$NEXTDATE
done
echo CYCLING COMPLETE
