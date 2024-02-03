#!/bin/bash
. $config_file
rundir=$work_dir/cycle/$time/vort2d
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
if [[ $time == $time_start ]]; then wait_for_module ../icbc; fi

echo running > stat

echo "  Running forecast for vort2d model..."

##load env if necessary
src_file=$script_dir/../config/env/$host/vort2d.src
if [[ -f $src_file ]]; then source $src_file; fi

touch run.log

$script_dir/job_submit.sh 1 1 0 python $script_dir/../models/vort2d/run.py $time >& run.log

nextdir=$work_dir/cycle/$next_time/vort2d
if [[ ! -d $nextdir ]]; then mkdir -p $nextdir; fi

watch_log run.log successfully 5 $rundir

cp -L ${next_time:0:8}_${next_time:8:2}.nc $nextdir/.

echo complete > stat

