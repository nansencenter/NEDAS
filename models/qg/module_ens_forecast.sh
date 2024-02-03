#!/bin/bash
. $config_file
rundir=$work_dir/cycle/$time/vort2d
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
if [[ $time == $time_start ]]; then wait_for_module ../perturb; fi
if [[ $time -gt $time_start ]]; then
    if $run_assim; then
        wait_for_module ../analysis
    else
        wait_for_module ../../$prev_time/vort2d
    fi
fi

echo running > stat

echo "  Running ensemble forecast for vort2d model..."

##load env if necessary
src_file=$script_dir/../config/env/$host/vort2d.src
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
    mv ../${time:0:8}_${time:8:2}_mem$m_id.nc .

    $script_dir/job_submit.sh 1 1 0 python $script_dir/../models/vort2d/run.py $time $m >& run.log &

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

