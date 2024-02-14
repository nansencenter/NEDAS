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
src_file=$script_dir/../config/env/$host/qg.src
if [[ -f $src_file ]]; then source $src_file; fi

tid=0
nt=$ntasks
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`
    if [[ ! -d $m_id ]]; then mkdir -p $m_id; fi
    touch $m_id/perturb.log

    ##run the model for member m
    cd $m_id

    ##make input.nml
    export input_type=spectral_m
    export random_seed=$RANDOM
    $script_dir/../models/qg/namelist.sh > input.nml

    $script_dir/job_submit.sh 1 1 $tid $script_dir/../models/qg/src/qg.exe . >& perturb.log &

    cd ..

    ##wait if ntasks processors are all in use
    tid=$((tid+1))
    if [[ $tid == $nt ]]; then tid=0; wait; fi

done
wait


##check output
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`

    if [[ ! -d ../qg/$m_id ]]; then mkdir -p ../qg/$m_id; fi

    watch_file $m_id/output.bin 1 $rundir
    cp $m_id/output.bin ../qg/$m_id/output_${time:0:8}_${time:8:2}.bin
done

echo complete > stat

