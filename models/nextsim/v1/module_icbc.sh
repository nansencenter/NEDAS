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
src_file=$script_dir/../config/env/$host/nextsim.v1.src
if [[ -f $src_file ]]; then source $src_file; fi

##initial condition comes from restart file from previous runs


#$script_dir/job_submit.sh 1 1 0 python $script_dir/../models/nextsim/v1/generate_ic.py $time >& icbc.log

###check output
#icfile=${time:0:8}_${time:8:2}.nc
#watch_file $icfile 5 $rundir

#if [[ ! -d ../nextsim.v1 ]]; then mkdir -p ../nextsim.v1; fi
#mv $icfile ../nextsim.v1/.

echo complete > stat

