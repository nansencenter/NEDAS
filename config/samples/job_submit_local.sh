#!/bin/bash
##this script runs parallel command on your host machine
##run as:
##   job_submit.sh nproc offset job_exe_cmd
nproc=$1
offset=$2
shift 2
exe_command=$@

##on local computer (laptop?) there is limited resources, just discard nproc and offset and run command directly
$exe_command

