#!/bin/bash
. $config_file

if [[ $time -gt $time_start ]]; then exit; fi

rundir=$work_dir/cycle/$time/icbc
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

echo running > stat

echo "  Generating initial and boundary conditions..."

##load env if necessary
src_file=$script_dir/../config/env/$host/vort2d.src
if [[ -f $src_file ]]; then source $src_file; fi

$script_dir/job_submit.sh 1 1 0 python $script_dir/../models/vort2d/generate_ic.py $time >& icbc.log

##check output
icfile=${time:0:8}_${time:8:2}.nc
watch_file $icfile 5 $rundir

if [[ ! -d ../vort2d ]]; then mkdir -p ../vort2d; fi
mv $icfile ../vort2d/.

echo complete > stat

