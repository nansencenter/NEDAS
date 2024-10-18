#!/bin/bash
##this script runs parallel command on your host machine
##run as:
##   job_submit.sh nproc offset exe_command
##arguments:
##   nproc = the total number of processors to run the job
##   offset = the starting processor index for the job
##            (job will use processor ids offset:offset+nproc)
##   job_exe_cmd = the run command for the job, including options etc.
##example:
##   on a host machine with 256 available processors, "job_submit.sh 128 0 model.exe"
## will run model.exe using the first 128 processors; "job_submit.sh 128 128 model.exe"
## will run another model.exe instance using the remaining 128 processors.

nproc=$1
offset=$2
shift 2
exe_command=$@

##on a super computer when allocated several compute nodes, the list of nodes and processor_per_node (ppn)
##shall be stored in an environment variable, the name of the variable depends on job scheduling system
##here we take SLURM as example (although srun command is handy to use in SLURM, so using mpiexec is not necessary)

##SLURM_TASKS_PER_NODE has format "128(x2)" for ppn=128 and nnode=2
ppn=$(echo $SLURM_TASKS_PER_NODE |awk -F'(' '{print $1}')

nnode=$(echo "($nproc+$ppn-1)/$ppn" |bc)
offset_node=$(echo "$offset/$ppn" |bc)

##SLURM_NODELIST has format "b[1183-1185,1192]" for nodes "b1183, b1184, b1185, b1192"
nodelist_avail=$(echo $SLURM_NODELIST |sed -e 's/^b\[//' -e 's/\]//' -e 's/,/ /g' |awk '{for(i=1;i<=NF;i++){if($i ~ /-/) {split($i,a,"-"); for(j=a[1];j<=a[2];j++) print "b"j} else print "b"$i}}')

nodelist=$(printf "%s\n" $nodelist_avail |head -n +$((offset_node+$nnode)) |tail -n $nnode)

##error handle: offset_node > nnode_avail
echo $nodelist

#echo mpiexec -v -np $nproc -machinefile <(echo '$nodelist') $exe_command

