#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_expt
#SBATCH --time=0-01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --partition=normal
#SBATCH --output=log/%x.out

nens=100
model=qg
config_file=config/samples/$model.yml

source $HOME/.bashrc
source $HOME/python.src

python -m NEDAS --config_file=$config_file --nproc=$SLURM_NTASKS --nens=$nens --work_dir=$SCRATCH/$model.n$nens
