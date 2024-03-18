#!/bin/bash
. $config_file

rundir=$work_dir/cycle/$time/icbc
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

echo running > stat

echo "  Generating initial and boundary conditions..."

##load env if necessary
src_file=$script_dir/../config/env/$host/topaz.v5.src
if [[ -f $src_file ]]; then source $src_file; fi

touch icbc.log

##initial condition comes from previous restart files
#if [[ $time == $time_start ]]; then
#    cp $data_dir/nextsim_ens/restart/{field,mesh}_${time:0:8}T${time:8:4}00Z.{bin,dat} .
#fi

##boundary condition is from era5 and convert to generic_ps_atm files
#mkdir GENERIC_PS_ATM
#fcst_days=$((forecast_period/24))  ##got forecast_period from top-level scripts
#for d in `seq 0 $fcst_days`; do
#    fcst_time=`advance_time $time $((d*24))`
#    cp $data_dir/generic_ps_atm/generic_ps_atm_${fcst_time:0:8}.nc GENERIC_PS_ATM/.
#done

#$script_dir/job_submit.sh 1 1 0 python $script_dir/../models/nextsim/v1/generate_ic.py $time >& icbc.log

###check output
#icfile=${time:0:8}_${time:8:2}.nc
#watch_file $icfile 5 $rundir

#if [[ ! -d ../nextsim.v1 ]]; then mkdir -p ../nextsim.v1; fi
#mv $icfile ../nextsim.v1/.

echo complete > stat

