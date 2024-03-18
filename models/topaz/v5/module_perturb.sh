#!/bin/bash
. $config_file

rundir=$work_dir/cycle/$time/perturb
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
wait_for_module ../icbc

echo running > stat

echo "  Generating perturbed ensemble members..."

##load env if necessary
src_file=$script_dir/../config/env/$host/topaz.v5.src
if [[ -f $src_file ]]; then source $src_file; fi

###TODO: perturb code here
###for now just cp perturbed ic and forcing
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`
    if [[ ! -d $m_id ]]; then mkdir -p $m_id; fi

    cd $m_id

    touch perturb.log

    ##initial condition from restart files
    #if [[ $time == $time_start ]]; then
    #    cp $data_dir/nextsim_ens/$m_id/restart/{field,mesh}_${time:0:8}T${time:8:4}00Z.{bin,dat} .
    #fi

    ##perturbed wind forcing
    #mkdir GENERIC_PS_ATM
    #fcst_days=$((forecast_period/24))
    #for d in `seq 0 $fcst_days`; do
    #    fcst_time=`advance_time $time $((d*24))`
    #    cp $data_dir/generic_ps_atm/generic_ps_atm_${fcst_time:0:8}.nc GENERIC_PS_ATM/.
    #done

    cd .. ##from $m_id
done

###check output
#for m in `seq 1 $nens`; do
#    m_id=`padzero $m 3`

#    icfile=${time:0:8}_${time:8:2}_mem$m_id.nc
#    watch_file $icfile 1 $rundir

#    mv $icfile ../topaz.v5/.
#done

echo complete > stat

