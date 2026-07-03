#!/bin/bash

#SBATCH --account=nn2993k
#SBATCH --job-name=run_expt
#SBATCH --time=0-00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=740G
#SBATCH --qos=devel
#SBATCH --partition=small
#SBATCH --output=log/%x.out

config_file=${PWD}/config.yml
container_path=${PROJECT}/nedas-tutorials_latest.sif

export APPTAINERENV_HYDRA_LAUNCHER=fork

apptainer exec \
    $container_path \
    python -m NEDAS --config_file=$config_file
