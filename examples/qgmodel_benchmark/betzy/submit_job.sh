#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=qg_bench
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --qos=devel
#SBATCH --ntasks-per-node=128
#SBATCH --partition=normal
#SBATCH --output=log/%x.out

source $HOME/.bashrc
source $HOME/python.src

cd /cluster/home/yingyue/code/NEDAS/examples/qgmodel_benchmark/betzy
python -m NEDAS --config_file=config.yml --nproc=$SLURM_NTASKS

