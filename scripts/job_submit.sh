#!/bin/bash

. $config_file

nnodes=$1
ncpus=$2
offset=$3
shift 3
exe_command=$@


###this is local computer (with a few processors):
if [[ $host == "localhost" ]]; then
    mpiexec -np $ncpus $exe_command
fi


###this works on Betzy
if [[ $host == "betzy" ]]; then
    offset_node=$((offset/$SLURM_NTASKS_PER_NODE))
    srun -N $nnodes -n $ncpus -r $offset_node --exact --unbuffered $exe_command
fi


###The manual solution, just in case:
#function get_nodefile {
#    job_nodelist=$1
#    nodefile=''
#    if [[ ${job_nodelist:1:1} == '[' ]]; then
#        pre=${job_nodelist:0:1}
#        for id in `echo $job_nodelist |awk -F '[][]' '{print $2}' |tr ',' ' '`; do
#            if [[ ${id:4:1} == '-' ]]; then
#                for j in `seq ${id:0:4} ${id:5:4}`; do
#                    nodefile=$nodefile' '$pre$j
#                done
#            else
#                nodefile=$nodefile' '$pre$id
#            fi
#        done
#    else
#        nodefile=$job_nodelist
#    fi
#    echo $nodefile |tr ' ' '\n'
#}
#get_nodefile $SLURM_JOB_NODELIST |head -n $((offset_node+$nnodes)) |tail -n $nnodes > nodefile
#mpirun -np $ncpus -machinefile nodefile $exe_command

