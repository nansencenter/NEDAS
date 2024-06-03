#!/bin/bash

nproc=$1
offset=$2
shift 2
exe_command=$@

if [ -z $SLURM_TASKS_PER_NODE ]; then
    mpiexec -n $nproc $exe_command
else
    ppn=$(echo $SLURM_TASKS_PER_NODE |awk -F'(' '{print $1}')

    nnode=$(echo "($nproc+$ppn-1)/$ppn" |bc)
    offset_node=$(echo "$offset/$ppn" |bc)

    srun -N $nnode -n $nproc -r $offset_node --exact --unbuffered $exe_command
fi

