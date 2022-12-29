#!/bin/bash
#SBATCH --account=nn2993k
#SBATCH --job-name=sample_fcst_error
#SBATCH --time=0-01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=preproc

cd $HOME/code/NEDAS

. $HOME/python.src; . $HOME/yp/bin/activate
set -a; . config/defaults; set +a

cd perturbation/sample_forecast_error

n_count=0
n_total=$SLURM_NTASKS

for n in `seq 1 60`; do
    #srun -n 1 -N 1 --exact python process_ECMWF.py $n &
    srun -n 1 -N 1 --exact python process_AROME.py $n &

    n_count=$((n_count+1))
    if [[ $n_count == $n_total ]]; then
        n_count=0
        wait
    fi
done
wait
