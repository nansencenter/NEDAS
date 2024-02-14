#!/bin/bash
. $config_file
rundir=$work_dir/cycle/$time/analysis
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
model_list=`echo "$state_def" |awk '{print $2}' |uniq`
for model in $model_list; do wait_for_module ../../$prev_time/$model; done

echo running > stat

echo "  Running data assimilation..."

##load env if necessary
src_file=$script_dir/../config/env/$host/python.src
if [[ -f $src_file ]]; then source $src_file; fi

$script_dir/job_submit.sh $nnodes $ntasks 0 python $script_dir/run_assim.py >& assim.log

##check output
watch_log assim.log successfully 2 $rundir

echo complete > stat

