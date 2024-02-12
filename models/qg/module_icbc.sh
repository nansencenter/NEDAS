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
src_file=$script_dir/../config/env/$host/qg.src
if [[ -f $src_file ]]; then source $src_file; fi

##make input.nml
export input_type=spectral_m
export random_seed=$RANDOM
$script_dir/../models/qg/namelist.sh > input.nml

$script_dir/job_submit.sh 1 1 0 $script_dir/../models/qg/src/qg.exe . >& icbc.log

##check output
icfile=output.bin
watch_file $icfile 1 $rundir

if [[ ! -d ../qg ]]; then mkdir -p ../qg; fi
cp $icfile ../qg/input.bin

echo complete > stat

