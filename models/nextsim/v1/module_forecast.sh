#!/bin/bash
. $config_file

rundir=$work_dir/cycle/$time/nextsim.v1
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
if [[ $time == $time_start ]]; then wait_for_module ../icbc; fi

echo running > stat

echo "  Running forecast with nextsim.v1 model..."

##load env if necessary
src_file=$script_dir/../config/env/$host/nextsim.v1.src
if [[ -f $src_file ]]; then source $src_file; fi

touch run.log

##link files for model run
rm -rf data
mkdir data
cd data
ln -fs $data_dir/BATHYMETRY/* .
ln -fs $data_dir/TOPAZ4/TP4DAILY_* .
mkdir -p GENERIC_PS_ATM
cp -L $data_dir/generic_ps_atm/generic_ps_atm_${time:0:8}.nc GENERIC_PS_ATM/.
cd ..
mkdir -p restart
cd restart
cp -L $data_dir/nextsim_ens/restart/{field,mesh}_${time:0:8}T${time:8:4}00Z.{bin,dat} .
cd ..
export NEXTSIM_DATA_DIR=`pwd`/data

##make the namelist
$script_dir/../models/nextsim/v1/namelist.sh > nextsim.cfg

##run the model
$script_dir/job_submit.sh $nnodes $ntasks 0 $code_dir/nextsim/model/bin/nextsim.exec --config-files=nextsim.cfg >& run.log

cd ..

nextdir=$work_dir/cycle/$next_time/nextsim.v1
if [[ ! -d $nextdir ]]; then mkdir -p $nextdir; fi

##collect output files, make a copy of forecast files to next cycle
watch_log $m_id/run.log successfully 5 $rundir

#mv $m_id/${next_time:0:8}_${next_time:8:2}_mem$m_id.nc .
#cp -L ${next_time:0:8}_${next_time:8:2}_mem$m_id.nc $nextdir/.

echo complete > stat


