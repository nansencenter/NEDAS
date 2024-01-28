#!/bin/bash
. $config_file

if [[ $time -gt $time_start ]]; then exit; fi

rundir=$work_dir/cycle/$time/perturb
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

echo running > stat

echo "  Generating perturbed ensemble members..."

##load env if necessary
src_file=$script_dir/../config/env/$host/nextsim.v1.src
if [[ -f $src_file ]]; then source $src_file; fi

#$script_dir/job_submit.sh 1 1 0 python $script_dir/../models/nextsim/v1/perturb_ic.py $time $nens >& perturb.log

#if [[ ! -d ../nextsim.v1 ]]; then mkdir -p ../nextsim.v1; fi

###check output
#for m in `seq 1 $nens`; do
#    m_id=`padzero $m 3`

#    icfile=${time:0:8}_${time:8:2}_mem$m_id.nc
#    watch_file $icfile 1 $rundir

#    mv $icfile ../nextsim.v1/.
#done

echo complete > stat

