#!/bin/bash
. $config_file

rundir=$work_dir/cycle/$time/nextsim
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
if [[ $time == $time_start ]]; then wait_for_module ../perturb; fi
if [[ $time -gt $time_start ]]; then
    if $run_assim; then
        wait_for_module ../analysis
    else
        wait_for_module ../../$prev_time/nextsim.v1
    fi
fi

echo running > stat

echo "  Running ensemble forecast for nextsim.v1 model..."

##load env if necessary
src_file=$script_dir/../config/env/$host/nextsim.v1.src
if [[ -f $src_file ]]; then source $src_file; fi

tid=0
nt=$ntasks
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`
    if [[ ! -d $m_id ]]; then mkdir -p $m_id; fi
    touch $m_id/run.log

    ##run the model for member m
    cd $m_id

    ##link files for model run
    rm -rf data
    mkdir data
    cd data
    ln -fs $data_dir/BATHYMETRY/* .
    ln -fs $data_dir/TOPAZ4/TP4DAILY_* .
    ln -fs /cluster/work/users/yingyue/data/GENERIC_PS_ATM/$casename/$mem GENERIC_PS_ATM
    cd ..
    export NEXTSIM_DATA_DIR=`pwd`/data

    mkdir -p restart
    cd restart
    ln -fs $SCRATCH/nextsim_ens_runs/init_run/restart/{field,mesh}_${DATE:0:8}T${DATE:8:4}00Z.{bin,dat} .
    cd ..

    ##make the namelist
    $script_dir/forecast_models/nextsim/namelist.sh > nextsim.cfg

    ##run the model
    $script_dir/job_submit.sh 1 128 $tid $code_dir/nextsim/model/bin/nextsim.exec --config-files=nextsim.cfg >& run.log &

    cd ..

    ##wait if ntasks processors are all in use
    tid=$((tid+1))
    if [[ $tid == $nt ]]; then tid=0; wait; fi

done
wait

nextdir=$work_dir/cycle/$next_time/vort2d
if [[ ! -d $nextdir ]]; then mkdir -p $nextdir; fi

##collect output files, make a copy of forecast files to next cycle
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`

    watch_log $m_id/run.log successfully 5 $rundir

    mv $m_id/${next_time:0:8}_${next_time:8:2}_mem$m_id.nc .
    cp -L ${next_time:0:8}_${next_time:8:2}_mem$m_id.nc $nextdir/.
done
wait

echo complete > stat


