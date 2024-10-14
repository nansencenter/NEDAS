#!/bin/bash
##this script runs parallel command on your host machine
##run as:
##   job_submit.sh nproc offset job_exe_cmd
##arguments:
##   nproc = the total number of processors to run the job
##   offset = the starting processor index for the job
##            (job will use processor ids offset:offset+nproc)
##   job_exe_cmd = the run command for the job, including options etc.

nproc=$1
offset=$2
shift 2
exe_command=$@

mpiexec -n $nproc $exe_command

