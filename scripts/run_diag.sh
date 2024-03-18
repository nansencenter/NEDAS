#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_diag
#SBATCH --time=0-04:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=preproc
#SBATCH --output=/cluster/home/yingyue/code/NEDAS/log/diag.uj

source ~/.bashrc
##other initial environment src code

#load configuration files, functions, parameters
export config_file=$HOME/code/NEDAS/config/qg_testcase
set -a; source $config_file; set +a

cd $script_dir
source util.sh

trap cleanup SIGINT SIGTERM


