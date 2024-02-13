#!/bin/bash
. $config_file
rundir=$work_dir/cycle/$time/qg
if [[ ! -d $rundir ]]; then mkdir -p $rundir; echo waiting > $rundir/stat; fi

cd $rundir
if [[ `cat stat` == "complete" ]]; then exit; fi

##check dependency
if [ $time == $time_start ]; then
    wait_for_module ../icbc
    wait_for_module ../perturb
fi
if $run_assim && [ $time -ge $time_assim_start ] && [ $time -le $time_assim_end ]; then wait_for_module ../analysis; fi
if [ $time -gt $time_start ]; then wait_for_module ../../$prev_time/qg; fi

echo running > stat

echo "  Running ensemble forecast for qg model..."

##run the model for each member
src_file=$script_dir/../config/env/$host/qg.src
if [[ -f $src_file ]]; then source $src_file; fi

tid=0
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`
    if [[ ! -d $m_id ]]; then mkdir -p $m_id; fi
    touch $m_id/run.log

    ##run the model for member m
    cd $m_id

    ##make input.nml
    export input_type=read
    $script_dir/../models/qg/namelist.sh > input.nml
    rm -f restart.nml
    ln -fs output_${time:0:8}_${time:8:2}.bin input.bin

    $script_dir/job_submit.sh 1 1 0 $script_dir/../models/qg/src/qg.exe . >& run.log &

    cd ..

    ##wait if ntasks processors are all in use
    tid=$((tid+1))
    if [[ $tid == $nt ]]; then tid=0; wait; fi

done
wait

##collect output files, make a copy of forecast files to next cycle
for m in `seq 1 $nens`; do
    m_id=`padzero $m 3`

    next_dir=$work_dir/cycle/$next_time/qg/$m_id
    if [[ ! -d $next_dir ]]; then mkdir -p $next_dir; fi

    watch_log $m_id/run.log "Calculation done" 1 $rundir

    watch_file $m_id/output.bin 1 $rundir
    mv $m_id/output.bin $m_id/output_${next_time:0:8}_${next_time:8:2}.bin

    cp $m_id/output_${next_time:0:8}_${next_time:8:2}.bin $next_dir/.

done

echo complete > stat

