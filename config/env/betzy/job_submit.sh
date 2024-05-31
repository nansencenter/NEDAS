#!/bin/bash

nproc=$1
offset=$2
shift 2
exe_command=$@

ppn=$(echo $SLURM_TASKS_PER_NODE |awk -F'(' '{print $1}')

nnode=$(echo "($nproc+$ppn)/$ppn" |bc)
offset_node=$(echo "$offset/$ppn" |bc)

echo srun -N $nnode -n $nproc -r $offset_node --exact --unbuffered $exe_command

echo error occur >&2
exit 1
