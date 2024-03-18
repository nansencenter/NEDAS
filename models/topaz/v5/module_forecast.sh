#!/bin/bash
. $config_file

rundir=$work_dir/cycle/$time/topaz.v5
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
wait_for_module ../icbc

echo running > stat

echo "  Running forecast with topaz.v5 model..."

##load env if necessary
src_file=$script_dir/../config/env/$host/topaz.v5.src
if [[ -f $src_file ]]; then source $src_file; fi

touch run.log

##link files for model run
#rm -rf data
#mkdir data
#cd data
#ln -fs $data_dir/BATHYMETRY/* .
#ln -fs $data_dir/TOPAZ4/TP4DAILY_* .
#ln -fs $work_dir/cycle/$time/icbc/GENERIC_PS_ATM .
#cd ..

#mkdir -p restart
#cd restart
#if [[ $time == $time_start ]]; then
#    ln -fs $work_dir/cycle/$time/icbc/{field,mesh}_${time:0:8}T${time:8:4}00Z.{bin,dat} .
#fi
#cd ..

##make the namelist
#$script_dir/../models/nextsim/v1/namelist.sh > nextsim.cfg

##run the model
#$script_dir/job_submit.sh $nnodes $ntasks 0 $code_dir/nextsim/model/bin/nextsim.exec --config-files=nextsim.cfg >& run.log

nextdir=$work_dir/cycle/$next_time/topaz.v5/restart
if [[ ! -d $nextdir ]]; then mkdir -p $nextdir; fi

##collect output files, make a copy of forecast file (prior) to
##next cycle directory to be updated to analysis file (posterior)
watch_log run.log "successfully" 1 $rundir

#mv restart/{field,mesh}_${next_time:0:8}T${next_time:8:4}00Z.{bin,dat} $nextdir/.

echo complete > stat


