#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=error_diag
#SBATCH --time=0-08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=preproc

cd $HOME/code/NEDAS

. $HOME/python.src; . $HOME/yp/bin/activate
set -a; . config/defaults; set +a

cd perturbation
python calc_spectrum.py /cluster/work/users/yingyue/perturb_param/sample_ECMWF 47 10 4 917
