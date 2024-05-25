#!/bin/bash

nnode=$1
nproc=$2
offset=$3
exe_command=$4

##localhost has no mpi, just run serial command
$exe_command

