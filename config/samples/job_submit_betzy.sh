#!/bin/bash
##this script runs parallel command on your host machine
##run as:
##   job_submit.sh nproc offset job_exe_cmd
##arguments:
##   nproc = the total number of processors to run the job
##   offset = the starting processor index for the job
##            (job will use processor ids offset:offset+nproc)
##   job_exe_cmd = the run command for the job, including options etc.
##example:
##   on a host machine with 256 available processors, "job_submit.sh 128 0 model.exe"
## will run model.exe using the first 128 processors; "job_submit.sh 128 128 model.exe"
## will run another model.exe instance using the remaining 128 processors.

##here is how its done on betzy:

nproc=$1
offset=$2
shift 2
exe_command=$@

if [ -z $SLURM_TASKS_PER_NODE ]; then
    ##running job from login node
    ##this is not suitable for large amount of processors
    if [ "$nproc" -gt 16 ] || [ "$offset" -gt 16 ]; then
        echo "You are on login node, don't run large parallel programs, try reducing nproc below 16"
        exit
    fi
    mpiexec -n $nproc $exe_command
else
    ppn=$(echo $SLURM_TASKS_PER_NODE |awk -F'(' '{print $1}')

    nnode=$(echo "($nproc+$ppn-1)/$ppn" |bc)
    offset_node=$(echo "$offset/$ppn" |bc)

    srun -N $nnode -n $nproc -r $offset_node --exact --unbuffered $exe_command
fi

