#!/bin/bash

nproc=$1
offset=$2
shift 2
exe_command=$@

offset_node=$((offset/$SLURM_TASKS_PER_NODE))

echo srun -n $nproc -r $offset_node --exact --unbuffered $exe_command
srun -n $nproc -r $offset_node --exact --unbuffered $exe_command

