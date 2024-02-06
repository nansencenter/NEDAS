#!/bin/bash
. $config_file

rundir=$work_dir/cycle/$time/nextsim.v1
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
wait_for_module ../icbc
wait_for_module ../perturb
if $run_assim && [ $time -ge $time_assim_start ] && [ $time -le $time_assim_end ]; then wait_for_module ../analysis; fi
if [ $time -gt $time_start ]; then wait_for_module ../../$prev_time/nextsim.v1; fi

echo running > stat

echo "  Running ensemble forecast for nextsim.v1 model..."

##load env if necessary
src_file=$script_dir/../config/env/$host/nextsim.v1.src
if [[ -f $src_file ]]; then source $src_file; fi

tid=0
nt=$nnodes
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`
    if [[ ! -d $m_id ]]; then mkdir -p $m_id; fi
    touch $m_id/run.log

    ##run the model for member m
    cd $m_id

    ##if finished skip this member
    if [ ! -z "`grep "Simulation done" run.log`" ]; then break; fi

    ##link files for model run
    rm -rf data
    mkdir data
    cd data
    ln -fs $data_dir/BATHYMETRY/* .
    ln -fs $data_dir/TOPAZ4/TP4DAILY_* .
    ln -fs $work_dir/cycle/$time/perturb/$m_id/GENERIC_PS_ATM .
    cd ..  ##from data

    mkdir -p restart
    cd restart
    if [[ $time == $time_start ]]; then
        ln -fs $work_dir/cycle/$time/perturb/$m_id/{field,mesh}_${time:0:8}T${time:8:4}00Z.{bin,dat} .
    fi
    cd ..  ##from restart
    export NEXTSIM_DATA_DIR=`pwd`/data

    ##make the namelist
    $script_dir/../models/nextsim/v1/namelist.sh > nextsim.cfg

    ##run the model
    $script_dir/job_submit.sh 1 $tasks_per_node $tid $code_dir/nextsim/model/bin/nextsim.exec --config-files=nextsim.cfg >& run.log &

    cd ..  ##from $m_id

    ##wait if ntasks processors are all in use
    tid=$((tid+1))
    if [[ $tid == $nt ]]; then tid=0; wait; fi

done
wait

##collect output files, make a copy of forecast files to next cycle
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`

    cd $m_id

    nextdir=$work_dir/cycle/$next_time/nextsim.v1/$m_id/restart
    if [[ ! -d $nextdir ]]; then mkdir -p $nextdir; fi

    watch_log run.log "Simulation done" 1 $rundir

    ##make a copy of the forecast to next_time (prior) to the next rundir (to be updated to posterior)
    cp restart/{field,mesh}_${next_time:0:8}T${next_time:8:4}00Z.{bin,dat} $nextdir/.

    ##link the intermediate output files to next dir for use by state_to_obs
    if $run_assim && [ $next_time -ge $time_assim_start ] && [ $next_time -le $time_assim_end ]; then
        ftime_min=`advance_time $next_time $obs_window_min`
        ftime_max=`advance_time $next_time $obs_window_max`

        rm -f {field,mesh}_final.{bin,dat}

        for fname in {field,mesh}_*.{bin,dat}; do
            tstr=`echo $fname |awk -F. '{print $1}' |awk -F_ '{print $NF}'`
            ftime=${tstr:0:8}${tstr:9:4}
            if [ $ftime -ge $ftime_min ] && [ $ftime -le $ftime_max ]; then
                #ln -fs $work_dir/cycle/$time/nextsim.v1/$m_id/$fname $nextdir/../.
                cp $fname $nextdir/../.
            fi
        done
    fi

    cd ..  ##from $m_id
done

echo complete > stat


