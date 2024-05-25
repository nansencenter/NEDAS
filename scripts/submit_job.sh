#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_cycle
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --qos=devel
#SBATCH --partition=normal

source $HOME/.bashrc
source $HOME/code/NEDAS/config/env/betzy/python.src

python run_cycle.py --config_file=$SCRATCH/qg/qg_testcase.yml


