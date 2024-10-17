#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=qg.n100
#SBATCH --time=0-01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --partition=normal
#SBATCH --output=log/%x.out
nens=100
model=qg

source $HOME/.bashrc
source $HOME/code/NEDAS/config/env/betzy/python.src

python $HOME/code/NEDAS/scripts/run_exp.py --config_file=$NEDAS/config/samples/$model.yml --nproc $SLURM_NTASKS --nproc_mem $SLURM_NTASKS --nens $nens --work_dir $SCRATCH/$model.n$nens

