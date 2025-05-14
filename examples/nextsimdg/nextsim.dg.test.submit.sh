#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_expt
#SBATCH --time=0-03:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=128
#SBATCH --partition=normal
#SBATCH --output=log/slurm-%j.out

source $HOME/.bashrc
source $HOME/python.src

NEDAS=/cluster/work/users/yingyue/nextsimdg/NEDAS
export PYTHONPATH=$NEDAS
cd $NEDAS
config_file=config/samples/nextsim.dg.betzy.yml

python scripts/run_expt.py --config_file=$config_file

