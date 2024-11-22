#!/bin/bash
##this script runs parallel command on your host machine
##run as:
##   job_submit.sh nproc offset job_exe_cmd
nproc=$1
offset=$2
shift 2
exe_command=$@

if command -v mpiexec > /dev/null 2>&1; then
    mpiexec -np $nproc $exe_command
else
    if [ $nproc -gt 1 ] || [ $offset -gt 1 ]; then
        echo "Warning: cannot find 'mpiexec', will only use 1 processor, discarding nproc=$nproc"
    fi
    $exe_command
fi
