#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=run_cycle
#SBATCH --time=0-01:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=128
#SBATCH --partition=normal
#SBATCH --output=log/%j.out

source $HOME/.bashrc
source $HOME/code/NEDAS/config/env/betzy/python.src

python $HOME/code/NEDAS/scripts/run_exp.py --config_file=$NEDAS/config/default.yml --nproc $SLURM_NTASKS --nproc_mem $SLURM_NTASKS --nens 500 --work_dir $SCRATCH/qg2

