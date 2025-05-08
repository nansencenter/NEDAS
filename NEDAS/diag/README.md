diagnostics for nextsim output, ensemble runs

1. Regrid the output (irregular mesh) to a uniform analysis grid (900x900 5km), which is easier for fft and obtaining spectra. This step is done by running
    python process_nextsim_state.py outdir var_id

outdir is where the nextsim field/mesh_<date>.bin/dat files are located. The ensemble members are stored in separate dirs 001, 002...

var_id = 0, 1, 2, 3 for sic, sit, siu/v and damage

the variables on analysis grid are saved as var_<date>.npy files

Note: This interpolate step can be potentially done during model run as well. Use statevector option in nextsim to output nc files. [statevector], regular, data/reference_grid.nc, variables=...

NEDAS/grid/gen_reference_grid.py generates a uniform reference_grid.nc for nextsim to read in. The python script make_uniform_grid.py generates the same grid x,y and save to grid.npy

The process_nextsim_state.py uses bamg interpFromMeshToMesh2dx to perform the interpolation, so it requires the pynextsim.NextsimBin to be correctly installed


2. Plot ensemble states.
    python plot_state.py var_id mem_id

generate figures showing nextsim variables on NorthPolarStereo projection. This requires cartopy to be correctly installed.


Note: how to run many python instances on betzy: use --exact option in srun

    ntasks=$SLURM_NTASKS
    n=0
    for m in `seq 0 40`; do
    for v in `seq 0 3`; do
        srun -N1 -n1 --exact python plot_state.py $v $m >& /dev/null &
        n=$((n+1))
        if [[ $n == $ntasks ]]; then
            n=0
            wait
        fi
    done
    done
    wait

3. Plot ensemble spread: plot_ens_sprd.py

view.html provides a web browser viewer for the ensemble states/spread

4. Plot power spectrum: plot_spectrum.py

5. Plot scale components derived by spatial filtering: plot_scale_component.py


