#!/bin/bash
. $config_file
rundir=$work_dir/cycle/$time/qg
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
if [[ $time == $time_start ]]; then wait_for_module ../icbc; fi

echo running > stat

echo "  Running forecast for qg model..."

##load env if necessary
src_file=$script_dir/../config/env/$host/qg.src
if [[ -f $src_file ]]; then source $src_file; fi

##make input.nml
export input_type=read
$script_dir/../models/qg/namelist.sh > input.nml
rm -f restart.nml
ln -fs output_${time:0:8}_${time:8:2}.bin input.bin

$script_dir/job_submit.sh 1 1 0 $script_dir/../models/qg/src/qg.exe . >& run.log

next_dir=$work_dir/cycle/$next_time/qg
if [[ ! -d $next_dir ]]; then mkdir -p $next_dir; fi

watch_log run.log "Calculation done" 1 $rundir

##check output
icfile=output.bin
watch_file $icfile 1 $rundir
cp $icfile $next_dir/output_${next_time:0:8}_${next_time:8:2}.bin

echo complete > stat

