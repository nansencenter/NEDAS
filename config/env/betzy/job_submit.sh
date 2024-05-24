#!/bin/bash

nnode=$1
nproc=$2
offset=$3
exe_command=$4

offset_node=$((offset/$SLURM_NTASKS_PER_NODE))

srun -N $nnode -n $nproc -r $offset_node --exact --unbuffered $exe_command

